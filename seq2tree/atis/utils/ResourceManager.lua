require('pl.stringx').import()
local class = require 'class'

local ResourceManager = torch.class('seq2tree.ResourceManager')

function ResourceManager:__init()
  self.n = 0
  self.resource_pool = {}
  self.occupied_dict = {}
  self.index_dict = {}
end

function ResourceManager:reset(n)
  self.n = n
  self.resource_pool = {}
  for i = 1, n do
    table.insert(self.resource_pool, i)
  end
  self.occupied_dict = {}
  self.index_dict = {}
end

function ResourceManager:allocate()
  assert((#self.resource_pool) > 0)
  local i = table.remove(self.resource_pool)
  self.occupied_dict[i] = true
  return i
end

function ResourceManager:allocate2(k1, k2)
  -- print (tostring(k1) .. ' ' .. tostring(k2))
  local i = self:allocate()
  if (self.index_dict[k1] == nil) then self.index_dict[k1] = {} end
  self.index_dict[k1][k2] = i
  return i
end

function ResourceManager:get2(k1, k2)
  if (self.index_dict[k1] == nil) or (self.index_dict[k1][k2] == nil) then
    return nil
  end
  local i = self.index_dict[k1][k2]
  if (self.occupied_dict[i] == true) then
    return i
  else
    return nil
  end
end

function ResourceManager:free(i)
  table.insert(self.resource_pool, i)
  self.occupied_dict[i] = nil
end
