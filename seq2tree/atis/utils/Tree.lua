require('pl.stringx').import()
local class = require 'class'

local Tree = torch.class('seq2tree.Tree')

function Tree:__init()
  self.parent = nil
  self.num_children = 0
  self.children = {}
end

function Tree:add_child(c)
  if class.istype(c, 'seq2tree.Tree') then
    c.parent = self
  end
  self.num_children = self.num_children + 1
  self.children[self.num_children] = c
end

function Tree:size()
  if self._size ~= nil then return self._size end
  local size = 1
  for i = 1, self.num_children do
    size = size + self.children[i]:size()
  end
  self._size = size
  return size
end

function Tree:children_vector()
  local r_list = {}
  for i = 1, self.num_children do
    if class.istype(self.children[i], 'seq2tree.Tree') then
      -- non-terminal symbol (4)
      table.insert(r_list, 4)
    else
      table.insert(r_list, self.children[i])
    end
  end
  return r_list
end

function Tree:to_string()
  local r_list = {}
  for i = 1, self.num_children do
    if class.istype(self.children[i], 'seq2tree.Tree') then
      table.insert(r_list, '( '..self.children[i]:to_string()..' )')
    else
      table.insert(r_list, tostring(self.children[i]))
    end
  end
  return (' '):join(r_list)
end

function Tree:to_list(form_manager)
  local r_list = {}
  for i = 1, self.num_children do
    if class.istype(self.children[i], 'seq2tree.Tree') then
      table.insert(r_list, form_manager:get_symbol_idx('('))
      local cl = self.children[i]:to_list(form_manager)
      for k = 1, #cl do table.insert(r_list, cl[k]) end
      table.insert(r_list, form_manager:get_symbol_idx(')'))
    else
      table.insert(r_list, self.children[i])
    end
  end
  return r_list
end
