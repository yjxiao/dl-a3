require 'torch'


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

function preprocess_data(raw_data, wordvector_table, opt)

    local data = torch.zeros(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs), 1, opt.inputDim+1, opt.inputLength)
    local labels = torch.zeros(opt.nClasses*(opt.nTrainDocs + opt.nTestDocs))

    -- use torch.randperm to shuffle the data, since it's ordered by class in the file
    local order = torch.randperm(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs))

    for i=1,opt.nClasses do
        for j=1,opt.nTrainDocs+opt.nTestDocs do
            local k = order[(i-1)*(opt.nTrainDocs+opt.nTestDocs) + j]

            local doc_size = 1

            local index = raw_data.index[i][j]
            -- standardize to all lowercase
            local document = ffi.string(torch.data(raw_data.content:narrow(1, index, 1)))

            -- break each review into words and compute the document average
            for word in document:gmatch("%S+") do
                if wordvector_table[word:lower():gsub("%p+", "")] then
		    ascii = word:byte()
	            if ascii >= 64 and ascii <= 90 then
		       data[{k, 1, 1, doc_size}] = 1
		    else
			data[{k, 1, 1, doc_size}] = 0
		    end

		    word = word:lower()

                    data[{k, 1, {2, opt.inputDim+1}, {doc_size}}] = (wordvector_table[word:gsub("%p+", "")])
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

function main()
    -- Configuration parameters
    opt = {}
    -- change these to the appropriate data locations
    opt.glovePath = "/scratch/courses/DSGA1008/A3/glove/glove.6B.50d.txt" -- path to raw glove data .txt file
    opt.dataPath = "/scratch/courses/DSGA1008/A3/data/train.t7b"
    -- word vector dimensionality
    opt.inputDim = 50
    opt.inputLength = 200
    -- nTrainDocs is the number of documents per class used in the training set, i.e.
    -- here we take the first nTrainDocs documents from each class as training samples
    -- and use the rest as a validation set.
    opt.nTrainDocs = 104000
    opt.nTestDocs = 26000
    opt.nClasses = 5

    print("Loading word vectors...")
    local glove_table = load_glove(opt.glovePath, opt.inputDim)

    print("Loading raw data...")
    local raw_data = torch.load(opt.dataPath)

    print("Computing document input representations...")
    local processed_data, labels = preprocess_data(raw_data, glove_table, opt)

    torch.save('/scratch/yx887/dl-a3/data_full_withCap.t7', processed_data)
    torch.save('/scratch/yx887/dl-a3/labels_full.t7', labels)
end

main()