import{TRAINING_DATA} from
'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js'

const INPUTS=TRAINING_DATA.inputs;
const OUTPUTS=TRAINING_DATA.outputs;
tf.util.shuffleCombo(INPUTS, OUTPUTS);

const TENSOR_INPUTS=tf.tensor2d(INPUTS);
const TENSOR_OUTPUTS=tf.tensor1d(OUTPUTS);

// function normalize(tensor, min, max){
//     const result=tf.tidy(function(){
//         const MIN_VALUES=min || tf.min(tensor,0);
//         const MAX_VALUES=max || tf.max(tensor,0);
//         const TENSOR_SUB_MIN_VALUES=tf.sub(tensor, MIN_VALUES);
//         const RANGE=tf.sub(MAX_VALUES, MIN_VALUES);
//         const NORMALIZE_VALUE=tf.div(TENSOR_SUB_MIN_VALUES, RANGE);
        
//         return {NORMALIZE_VALUE, MIN_VALUES, MAX_VALUES};


//     });
//     return result;
// }

//rewriting a noralize function
function normalize(tensor, min, max){
    const result=tf.tidy(function(){
        const MIN_VALUE=min|| tf.min(tensor, 0);
        const MAX_VALUE=max || tf.max(tensor, 0);
        const TENSOR_SUB_MIN_VALUES=tf.sub(tensor, MIN_VALUES);
        const RANGE=tf.sub(MAX_VALUE,MIN_VALUE);
        const NORMALIZE_VALUE=tf.div(TENSOR_SUB_MIN_VALUES, RANGE);
        return {NORMALIZE_VALUE,MIN_VALUE,MAX_VALUE};
    });
    return result;
}
                  

const FEATURE_RESULTS= normalize(TENSOR_INPUTS);

console.log("Normalized Values: ");
FEATURE_RESULTS.NORMALIZE_VALUE.print();

console.log("Minimum Values: ");
FEATURE_RESULTS.MIN_VALUES.print();

console.log("Maximum values: ");
FEATURE_RESULTS.MAX_VALUES.print();

TENSOR_INPUTS.dispose();

const model=tf.sequential();
model.add(tf.layers.dense({inputShape: [2], units:1}));
model.summary();
train();

async function train(){
    const LEATNING_RATE=0.05;

    model.compile({
        optimizer:tf.train.sgd(LEATNING_RATE),
        loss:"meanSquaredError",
    });

    let result=await model.fit(FEATURE_RESULTS.NORMALIZE_VALUE,TENSOR_OUTPUTS, {
        validationSplit:0.15,
        shuffle:true,
        batchSize:64,
        epoch:10,

    });
    TENSOR_OUTPUTS.dispose();
    FEATURE_RESULTS.NORMALIZE_VALUE.dispose();

    console.log("Average error loss: " + Math.sqrt(result.history.loss[result.history.loss.length-1]));
    console.log("Average validation error loss: " + Math.sqrt(result.history.val_loss[result.history.val_loss.length-1]));

    evaluate();
}

function evaluate(){
    tf.tidy(function(){
        let new_inputs=normalize(tf.tensor2d([[750,1]]), FEATURE_RESULTS.MIN_VALUES, FEATURE_RESULTS.MAX_VALUES);
        let output=model.predict(new_inputs.NORMALIZE_VALUE);
        output.print();
    });
    FEATURE_RESULTS.MIN_VALUES.dispose();
    FEATURE_RESULTS.MAX_VALUES.dispose();
    model.dispose();
    
    console.log(tf.memory().numTensors);

}



