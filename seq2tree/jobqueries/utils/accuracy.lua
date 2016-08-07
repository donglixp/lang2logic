require 'utils/SymbolsManager'
include "../utils/utils.lua"
local class = require 'class'

function is_all_same(c1,c2)
  if #c1 == #c2 then
    local all_same = true
    for j = 1, #c1 do
      if c1[j] ~= c2[j] then
        all_same = false
        break
      end
    end
    return all_same
  else
    return false
  end
end

function compute_accuracy(candidate_list, reference_list)
  if #candidate_list ~= #reference_list then
    print(string.format("LENGTH: #candidate_list(%d) ~= #reference_list(%d)", #candidate_list, #reference_list))
  end

  local len = math.min(#candidate_list, #reference_list)
  
  local c = 0
  for i = 1, len do
    if is_all_same(candidate_list[i], reference_list[i]) then
      c = c + 1
    end
  end

  return c / len
end

function pairsByKeys (t, f)
  local a = {}
  for n in pairs(t) do table.insert(a, n) end
  table.sort(a, f)
  local i = 0      -- iterator variable
  local iter = function ()   -- iterator function
    i = i + 1
    if a[i] == nil then return nil
    else return a[i], t[a[i]]
    end
  end
  return iter
end

function norm_tree(r_list, form_manager)
  local q = {[1] = convert_to_tree(r_list, 1, #r_list, form_manager)}
  local head = 1
  while head <= #q do
    local t = q[head]
    -- if this level is ``and'' operator
    if (t.children[1] == form_manager:get_symbol_idx('and')) or (t.children[1] == form_manager:get_symbol_idx('or')) then
      -- sort the following subchildren
      local k = {}
      for i = 2, #t.children do
        if class.istype(t.children[i], 'seq2tree.Tree') then
          k[t.children[i]:to_string()] = i
        else
          k[tostring(t.children[i])] = i
        end
      end
      local sorted_t_dict = {}
      local j = 2
      for child_str, child_index in pairsByKeys(k) do
        sorted_t_dict[j] = t.children[child_index]
        j=j+1
      end
      for i = 2, #t.children do
        t.children[i] = sorted_t_dict[i]
      end
    end
    -- add children to q
    for i = 1, #t.children do
      if class.istype(t.children[i], 'seq2tree.Tree') then
        table.insert(q, t.children[i])
      end
    end
    head = head + 1
  end

  return q[1]
end

function compute_tree_accuracy(candidate_list_, reference_list_, form_manager)
  local candidate_list = {}
  for i = 1, #candidate_list_ do
    table.insert(candidate_list, norm_tree(candidate_list_[i], form_manager):to_list(form_manager))
  end

  local reference_list = {}
  for i = 1, #reference_list_ do
    table.insert(reference_list, norm_tree(reference_list_[i], form_manager):to_list(form_manager))
  end

  return compute_accuracy(candidate_list, reference_list)
end
