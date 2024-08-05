#!/usr/bin/env python3

# Copyright (C) 2024 Grzegorz Kociołek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWA

import sys
import re
import argparse
import scipy
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel

# TODO:
# - Optimize processing of inputs smaller than the window length
# - Fix averaging of values in the remainder window
# - fix calculation of importance scores near the edges when the input is windowed

# Importance analyzer pipeline
class AnalyzerPipeline:
  OVERLAP_COMPOSITE_MODES = ["mean", "sum"]

  def __init__(self, tokenizer, model, lines, flags = {}):
    if not isinstance(lines, list):
      raise ValueError("lines parameter must be a list")

    self.tokenizer = tokenizer
    self.model = model
    self.lines = lines
    self.flags = flags
    self.output_importance = None
    self._commit_window_info()
    self._check_overlap_composite()

  def _commit_window_info(self):
    window_overlap = self.get_flag('window_overlap') or 0.5
    self.window_size = self.model.embeddings.position_embeddings.weight.shape[0] - 4
    self.window_offset = int((1.0 - window_overlap) * self.window_size)

  def _check_overlap_composite(self):
    composite_mode = self.get_flag('overlap_composite')
    if composite_mode != None and composite_mode not in AnalyzerPipeline.OVERLAP_COMPOSITE_MODES:
      raise ValueError(f'Invalid value for the "overlap_composite" flag (supported modes: {str.join(",",AnalyzerPipeline.OVERLAP_COMPOSITE_MODES)})')

  def __call__(self):
    if self.output_importance == None:
      self._encode_lines()
      self._fold_tokens()
      self._batch_evaluate()
      self._calculate_token_importance()
      self._unfold_importance()
      self._obtain_line_importance()

    return self.output_importance

  def _encode_lines(self):
    self.enc_lines = [self.tokenizer.encode(line + "\n", add_special_tokens=False) for line in self.lines]
    self.joint_tokens = torch.tensor([token for line in self.enc_lines for token in line])

  # Get sliding windows of the tokens
  def _fold_tokens(self):
    offset_range = range(0,max(len(self.joint_tokens) - self.window_size,0) + self.window_offset,self.window_offset)
    n_windows = len(offset_range)

    r_tok_windows = torch.full(size=(n_windows,self.window_size+2),
                               fill_value=self.tokenizer.pad_token_id)

    for i, offset in enumerate(offset_range):
      window = self.joint_tokens[offset:offset+self.window_size]
      r_tok_windows[i,1:len(window)+1] = window
      r_tok_windows[i,len(window)+1] = self.tokenizer.eos_token_id

    r_tok_windows[:,0] = self.tokenizer.bos_token_id
    self.token_windows = r_tok_windows

  def _batch_evaluate(self):
    attention_mask = torch.ones(size=self.token_windows.size())
    attention_mask.masked_fill_(self.token_windows == self.tokenizer.pad_token_id, 0)

    print(self.token_windows.shape)
    print(attention_mask.shape)

    inputs = { 'input_ids': self.token_windows,
              'attention_mask': attention_mask }

    # Forward pass to get the outputs and attention weights
    with torch.no_grad():
        outputs = self.model(**inputs, output_attentions=True)

    self.batched_output = outputs

  def _calculate_token_importance(self):
    last_layer = self.batched_output.attentions[-1]

    # Sum over heads
    attention_tensor = last_layer.sum(dim=1)

    for matrix in attention_tensor:
      matrix.fill_diagonal_(0)

    self.tok_imp_vec = attention_tensor.sum(dim=-2)

  def _unfold_importance(self):
    n_tokens = len(self.joint_tokens)
    offset_range = range(0,max(n_tokens - self.window_size,0) + self.window_offset,self.window_offset)

    attention_dtype = self.tok_imp_vec.dtype
    coef_vec = torch.zeros(size=(n_tokens,),dtype=attention_dtype)

    for offset in offset_range:
      coef_vec[offset:offset+self.window_size] += 1

    raw_windows = self.tok_imp_vec[:,1:-1]
    unfolded_vec = torch.zeros(size=(n_tokens,),dtype=attention_dtype)
    for i, offset in enumerate(offset_range):
      n_space = len(unfolded_vec[offset:offset+self.window_size])
      unfolded_vec[offset:offset+self.window_size] += raw_windows[i,:n_space]

    composite_mode = self.get_flag('overlap_composite')
    if composite_mode == 'add':
      pass
    else: # Assuming mean by default
      unfolded_vec /= coef_vec

    self.unfolded_vec = unfolded_vec

  def _obtain_line_importance(self):
    out_imp = torch.empty(size=(len(self.enc_lines),), dtype=self.unfolded_vec.dtype)

    line_offsets = [0] * (len(self.enc_lines)+1)
    for i, token_line in enumerate(self.enc_lines):
      line_offsets[i+1] = line_offsets[i] + len(token_line)


    for i, _ in enumerate(self.enc_lines):
      out_imp[i] = self.unfolded_vec[line_offsets[i]:line_offsets[i+1]].sum()


    if self.get_flag('mean_line_importance'):
      out_imp /= torch.tensor([len(token_line) for token_line in self.enc_lines])

    self.output_importance = out_imp


  def get_flag(self, flag):
    try:
      return self.flags[flag]
    except KeyError:
      return None


# Returns the model and the encoder
def load_model(name, device, force = False):
  tokenizer = RobertaTokenizer.from_pretrained(name)
  model = RobertaModel.from_pretrained(name)
  if force:
    model = model.to(device)
  else:
    try:
      model = model.to(device)
    except RuntimeError as e:
      print(f"warning: falling back to cpu as the model cannot be forwarded to device '{device}': '{e}'")
      pass

  model.eval()

  return [tokenizer, model]


# Create an ordered index map of non-empty lines
def create_line_map(lines):
  rxp = re.compile(r'^\s*$')
  return list(filter(lambda i: not re.match(rxp, lines[i]), range(len(lines))))

# Return a color escape based on intensity <0.0, 1.0>
def ascii_intensity_to_color_escape(intensity):
  norm = np.clip(intensity, 0.0, 1.0) #* 0.666 + 0.2
  color = [norm, norm/3, 1.0-norm]
  r, g, b = [int(x * 255) for x in color]
  return f"\x1b[48;2;{r};{g};{b}m"

ASCII_ESCAPE_RESET = "\x1b[0m"

def main():
  parser = argparse.ArgumentParser(
      description="Calculate the importance score for each line of text",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('file', metavar='FILE', type=str)
  parser.add_argument('-m', metavar='MODEL', type=str, default='microsoft/codebert-base', help="The model")
  parser.add_argument('-e', action='store_true', help="Keep empty lines")
  parser.add_argument('-S', action='store_true', help="Calculate the softmax of importance scores")
  parser.add_argument('-M', action='store_true', help="Use mean attention instead of total attention per line")
  parser.add_argument('--device', type=str, default='cuda', help="Device to use for model inference (falls back to cpu)")
  parser.add_argument("--force-device", action='store_true', help="Force the specified device and fail if it cannot be used")
  parser.add_argument('--window-overlap', metavar="PERCENT", type=float, default=50, help="Percent of overlap between token windows")
  parser.add_argument('--rank-color', action='store_true', help="Color based on rank position instead of absolute importance value")

  args = parser.parse_args()

  with open(args.file, "r") as f:
    src_code = f.read()

  src_lines = src_code.split("\n")

  if args.e:
    line_map = list(range(len(src_lines)))
  else:
    line_map = create_line_map(src_lines)

  filtered_lines = [src_lines[i] for i in line_map]

  tokenizer, model = load_model(args.m, args.device, force = args.force_device)

  flags = {
      'mean_line_importance': args.M
  }

  analyzer = AnalyzerPipeline(tokenizer, model, filtered_lines, flags)
  line_scores = analyzer()

  line_scores -= line_scores.min()
  line_scores /= line_scores.max()

  if args.S:
    line_scores = scipy.special.softmax(line_scores)

  # Save a ranking position of lines
  sorted_idx = list(sorted(range(len(line_scores)), key=lambda i: line_scores[i]))
  line_info = [None] * len(src_lines)

  for i, score in enumerate(line_scores):
    line_info[line_map[i]] = [score, None]

  for i, idx in enumerate(sorted_idx):
    line_info[line_map[idx]][1] = i

  # Print the importance scores
  for line, info in zip(src_lines, line_info):
    if info == None:
      print(f"-: {line}")
    else:
      color_intensity = (info[1] + 1) / len(line_scores) if args.rank_color else info[0]
      col_escape = ascii_intensity_to_color_escape(color_intensity)
      print(f"{col_escape}{info[0]:.4f}{ASCII_ESCAPE_RESET}: {line}")


if __name__ == '__main__':
  main()


