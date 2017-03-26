require 'torch'
require 'nn'
require 'cudnn'
require 'paths'
require 'bnn'


modelname = 'human_pose_binary'
local net = torch.load('models/human_pose_binary.t7'):float()
-- print(net)

-- local img = image.load(fileLists[i].image)
-- local originalSize = img:size()

-- img = utils.crop(img, fileLists[i].center, fileLists[i].scale, 256)
-- img = img:cuda():view(1,3,256,256)
local input = torch.rand(1,3,256,256)

generateGraph = require 'optnet.graphgen'

-- visual properties of the generated graph
-- follows graphviz attributes
graphOpts = {
displayProps =  {shape='ellipse',fontsize=14, style='solid'},
nodeData = function(oldData, tensor)
  return oldData .. '\n' .. 'Size: '.. tensor:numel()
end
}

g = generateGraph(net, input, graphOpts)

graph.dot(g,modelname,modelname)


-- -- some handy models are defined in optnet.models
-- -- like alexnet, googlenet, vgg and resnet
-- models = require 'optnet.models'
-- modelname = 'googlenet'
-- net, input = models[modelname]()

-- generateGraph = require 'optnet.graphgen'

-- -- visual properties of the generated graph
-- -- follows graphviz attributes
-- graphOpts = {
-- displayProps =  {shape='ellipse',fontsize=14, style='solid'},
-- nodeData = function(oldData, tensor)
--   return oldData .. '\n' .. 'Size: '.. tensor:numel()
-- end
-- }

-- g = generateGraph(net, input, graphOpts)

-- graph.dot(g,modelname,modelname)