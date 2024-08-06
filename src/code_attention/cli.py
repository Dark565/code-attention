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

import re
import warnings
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from .analyzer import AnalyzerPipeline

# Returns the model and the encoder
def load_model(name, device, force = False):
  tokenizer = AutoTokenizer.from_pretrained(name)
  model = AutoModel.from_pretrained(name)

  if force:
    model = model.to(device)
  else:
    try:
      model = model.to(device)
    except RuntimeError as e:
      print(f"warning: falling back to cpu as the model cannot be forwarded to device '{device}': '{e}'")
      model = model.to("cpu")

  # Some models (e.g GPT2) may not have the built-in pad token. Map it to the EOS token.
  if tokenizer.pad_token == None:
    tokenizer.pad_token = tokenizer.eos_token

  model.eval()

  return [tokenizer, model]


# Create an ordered index map of non-empty lines
def create_line_map(lines):
  rxp = re.compile(r'^\s*$')
  return list(filter(lambda i: not re.match(rxp, lines[i]), range(len(lines))))

# Return a color escape based on intensity <0.0, 1.0>
def ascii_intensity_to_color_escape(intensity):
  norm = torch.tensor(intensity).clip(0.0, 1.0).item() #* 0.666 + 0.2
  color = [norm, norm/3, 1.0-norm]
  r, g, b = [int(x * 255) for x in color]
  return f"\x1b[48;2;{r};{g};{b}m"

ASCII_ESCAPE_RESET = "\x1b[0m"

def float_range(min_v, max_v):
  def check_range(v):
    res = float(v)
    if not (res >= min_v and res <= max_v):
      raise argparse.ArgumentTypeError(f"float must be in range: <{min_v}, {max_v}>")

    return res

  return check_range


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
  parser.add_argument('--window-overlap', metavar="PERCENT", type=float_range(0.0,100.0), default=50, help="Percent of overlap between token windows")
  parser.add_argument('--overlap-composite', metavar="METHOD", type=str, choices=AnalyzerPipeline.OVERLAP_COMPOSITE_MODES, default='mean',
                      help='The method for combining overlapping windows')

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
      'mean_line_importance': args.M,
      'window_overlap': args.window_overlap / 100.0,
      'overlap_composite': args.overlap_composite
  }


  analyzer = AnalyzerPipeline(tokenizer, model, filtered_lines, flags)
  line_scores = analyzer()

  line_scores -= line_scores.min()
  line_scores /= line_scores.max()

  if args.S:
    line_scores = line_scores.softmax(dim=0)

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


