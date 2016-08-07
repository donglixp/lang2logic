require('pl.stringx').import()
require 'torch'

local SymbolsManager = torch.class('SymbolsManager')

function SymbolsManager:__init(whether_add_special_tags)
  self.symbol2idx = {}
  self.idx2symbol = {}
  self.vocab_size = 0
  self.whether_add_special_tags = whether_add_special_tags

  if whether_add_special_tags then
    -- start symbol (1)
    self:add_symbol('<S>')
    -- end symbol (2)
    self:add_symbol('<E>')
    -- UNK symbol (3)
    self:add_symbol('<U>')
  end
end

function SymbolsManager:add_symbol(s)
  if self.symbol2idx[s] == nil then
    self.vocab_size = self.vocab_size + 1
    self.symbol2idx[s] = self.vocab_size
    self.idx2symbol[self.vocab_size] = s
  end
  return self.symbol2idx[s]
end

function SymbolsManager:get_symbol_idx(s)
  if self.symbol2idx[s] == nil then
    if self.whether_add_special_tags then
      return self.symbol2idx['<U>']
    else
      return 0
    end
  end
  return self.symbol2idx[s]
end

function SymbolsManager:get_idx_symbol(idx)
  if self.idx2symbol[idx] == nil then
  	return '<U>'
  end
  return self.idx2symbol[idx]
end

function SymbolsManager:init_from_file(fn, min_freq, max_vocab_size)
  print('loading vocabulary file: ' .. fn)
  local f = torch.DiskFile(fn,'r',true)
  f:clearError()
  local rawdata = f:readString('*l')
  while (not f:hasError()) do
    local l_list = rawdata:strip():split('\t')
    local c = tonumber(l_list[2])
    if c >= min_freq then
      self:add_symbol(l_list[1])
    end
    if self.vocab_size >= max_vocab_size then
      break
    end
    -- read next line
    rawdata = f:readString('*l')
  end
  f:close()
end

function SymbolsManager:get_symbol_idx_for_list(l)
  local r = {}
  for i = 1, #l do
    table.insert(r, self:get_symbol_idx(l[i]))
  end
  return r
end
