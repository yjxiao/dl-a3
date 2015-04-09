require 'torch'
require 'cunn'
require 'optim'

ffi = require('ffi')

--- Parses and loads the GloVe word vectors into a hash table:
-- glove_table['word'] = vector
function load_glove(path, inputDim)
    
    local glove_file = io.open(path)
    local glove_table = {}

    local line = glove_file:read("*l")
    while line do
        -- read the GloVe text file one line at a time, break at EOF
        local i = 1
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if i == 1 then
                -- word comes first in each line, so grab it and create new table entry
                word = entry:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
                if string.len(word) > 0 then
                    glove_table[word] = torch.zeros(inputDim, 1) -- padded with an extra dimension for convolution
                else
                    break
                end
            else
                -- read off and store each word vector element
                glove_table[word][i-1] = tonumber(entry)
            end
            i = i+1
        end
        line = glove_file:read("*l")
    end
    
    return glove_table
end

--- Here we simply encode each document as a fixed-length vector 
-- by computing the unweighted average of its word vectors.
-- A slightly better approach would be to weight each word by its tf-idf value
-- before computing the bag-of-words average; this limits the effects of words like "the".
-- Still better would be to concatenate the word vectors into a variable-length
-- 2D tensor and train a more powerful convolutional or recurrent model on this directly.
function preprocess_data(raw_data, wordvector_table, opt)
    
    local data = torch.zeros(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs), 1, opt.inputDim, opt.inputLength)
    local labels = torch.zeros(opt.nClasses*(opt.nTrainDocs + opt.nTestDocs))
    
    -- use torch.randperm to shuffle the data, since it's ordered by class in the file
    local order = torch.randperm(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs))
    
    for i=1,opt.nClasses do
        for j=1,opt.nTrainDocs+opt.nTestDocs do
            local k = order[(i-1)*(opt.nTrainDocs+opt.nTestDocs) + j]
            
            local doc_size = 1
            
            local index = raw_data.index[i][j]
            -- standardize to all lowercase
            local document = ffi.string(torch.data(raw_data.content:narrow(1, index, 1))):lower()
            
            -- break each review into words and compute the document average
            for word in document:gmatch("%S+") do
                if wordvector_table[word:gsub("%p+", "")] then
                    data[{k, 1, {}, {doc_size}}] = (wordvector_table[word:gsub("%p+", "")])
		    if doc_size == opt.inputLength then
		        break
		    end
                    doc_size = doc_size + 1
                end
            end

            labels[k] = i
        end
    end

    return data, labels
end

function train_model(model, criterion, data, labels, test_data, test_labels, opt)

    model:cuda()
    criterion:cuda()
    local cudabatch = torch.zeros(opt.minibatchSize, 1, data:size(3), data:size(4)):cuda()
    local cudabatch_labels = torch.zeros(opt.minibatchSize):cuda()

    parameters, grad_parameters = model:getParameters()
    
    -- optimization functional to train the model with torch's optim library
    local function feval(x) 
        cudabatch[{}] = data:sub(opt.idx, opt.idx+opt.minibatchSize-1)
        cudabatch_labels[{}] = labels:sub(opt.idx, opt.idx+opt.minibatchSize-1)
        
        model:training()
        local minibatch_loss = criterion:forward(model:forward(cudabatch), cudabatch_labels)
        model:zeroGradParameters()
        model:backward(cudabatch, criterion:backward(model.output, cudabatch_labels))
	confusion:batchAdd(model.output, cudabatch_labels)

        return minibatch_loss, grad_parameters
    end
    
    for epoch=1,opt.nEpochs do
        local order = torch.randperm(opt.nBatches) -- not really good randomization
        for batch=1,opt.nBatches do
	    xlua.progress(batch, opt.nBatches)
            opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
            optim.sgd(feval, parameters, opt)
        end
	
	print(confusion)
	confusion:zero()

    end
end

function test_model(model, data, labels)
    
    model:evaluate()
    local batchSize = 100
    local nBatches = data:size(1) / 100
    local cudabatch = torch.zeros(batchSize, 1, data:size(3), data:size(4)):cuda()
    local cudabatch_labels = torch.zeros(batchSize):cuda()

    confusion:zero()
    for batch=1, nBatches do
        xlua.progress(batch, nBatches)
        cudabatch[{}] = data:sub((batch-1)*batchSize+1, batch*batchSize)
        cudabatch_labels[{}] = labels:sub((batch-1)*batchSize+1, batch*batchSize)
	local pred = model:forward(cudabatch)
	confusion:batchAdd(pred, cudabatch_labels)
    end
    confusion:updateValids()

    return confusion.totalValid
end

function main()

    -- Configuration parameters
    opt = {}
    -- change these to the appropriate data locations
    opt.glovePath = "/scratch/courses/DSGA1008/A3/glove/glove.6B.50d.txt" -- path to raw glove data .txt file
    opt.dataPath = "/scratch/courses/DSGA1008/A3/data/train.t7b"
    -- word vector dimensionality
    opt.inputDim = 50 
    opt.inputLength = 120
    -- nTrainDocs is the number of documents per class used in the training set, i.e.
    -- here we take the first nTrainDocs documents from each class as training samples
    -- and use the rest as a validation set.
    opt.nTrainDocs = 20000
    opt.nTestDocs = 5000
    opt.nClasses = 5
    -- SGD parameters - play around with these
    opt.nEpochs = 5
    opt.minibatchSize = 128
    opt.nBatches = math.floor(opt.nTrainDocs / opt.minibatchSize)
    opt.learningRate = 0.1
    opt.learningRateDecay = 0.001
    opt.momentum = 0.1
    opt.idx = 1

    print("Loading word vectors...")
    local glove_table = load_glove(opt.glovePath, opt.inputDim)
    
    print("Loading raw data...")
    local raw_data = torch.load(opt.dataPath)
    
    print("Computing document input representations...")
    local processed_data, labels = preprocess_data(raw_data, glove_table, opt)
    
    -- split data into makeshift training and validation sets
    local trainSize = opt.nClasses * opt.nTrainDocs
    local testSize = opt.nClasses * opt.nTestDocs
    local training_data = processed_data:sub(1, trainSize)
    local training_labels = labels:sub(1, trainSize)
    local test_data = processed_data:sub(trainSize+1, trainSize+testSize)
    local test_labels = labels:sub(trainSize+1, trainSize+testSize)

    -- initialize confusion matrix
    classes = {'1', '2', '3', '4', '5'}
    confusion = optim.ConfusionMatrix(classes)

    -- construct model:
    ninputs = 1
    nstates = {20}
    noutputs = 5
    filtsizeW = 10	-- filter size across words
    filtsizeH = 10	-- filter size across word vector dimensions
    poolsize = 3	-- pooling size across word vector dimensions

    model = nn.Sequential()

    model:add(nn.SpatialConvolutionMM(ninputs, nstates[1], filtsizeW, filtsizeH))
    model:add(nn.SpatialMaxPooling(111, poolsize, 1, 1))

    model:add(nn.Reshape(nstates[1]*39, true))
    model:add(nn.Linear(nstates[1]*39, 5))
    model:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()
   
    train_model(model, criterion, training_data, training_labels, test_data, test_labels, opt)
    local results = test_model(model, test_data, test_labels)
    print(results)
end

main()
