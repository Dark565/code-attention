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

import torch

# TODO:
# - Optimize processing of inputs smaller than the window length
# - fix calculation of importance scores near the edges when the input is windowed
# - Add type annotations and better error handling

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

  def _check_model(self):
    if self.tokenizer.pad_token_id == None:
      raise ValueError('Selected tokenizer doesn\'t have the pad token. It should be added before instantiating this class')

  def _commit_window_info(self):
    window_overlap = self.get_flag('window_overlap') or 0.5
    if window_overlap < 0.0:
      raise ValueError("Invalid window_overlap value")

    if window_overlap > 1.0:
      warnings.warn("The 'window_overlap' option is greater than 1")


    self.window_size = self.tokenizer.model_max_length - 2
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

  # Convert lines of text to token id lists
  def _encode_lines(self):
    self.enc_lines = [self.tokenizer.encode(line + "\n", add_special_tokens=False) for line in self.lines]
    self.joint_tokens = torch.tensor([token for line in self.enc_lines for token in line])

  # Extract windows out of the full token sequence
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

  # Batch token windows through the model
  def _batch_evaluate(self):
    attention_mask = torch.ones(size=self.token_windows.size())
    attention_mask.masked_fill_(self.token_windows == self.tokenizer.pad_token_id, 0)

    inputs = { 'input_ids': self.token_windows,
              'attention_mask': attention_mask }

    with torch.no_grad():
        outputs = self.model(**inputs, output_attentions=True, output_hidden_states=False)

    self.batched_output = outputs

  # Calculate token importance in windows
  def _calculate_token_importance(self):
    last_layer = self.batched_output.attentions[-1]

    # Sum over heads
    attention_tensor = last_layer.sum(dim=1)

    for matrix in attention_tensor:
      matrix.fill_diagonal_(0)

    self.tok_imp_vec = attention_tensor.sum(dim=-2)

  # Combine all windows
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

  # Obtain the importance values per lines
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
