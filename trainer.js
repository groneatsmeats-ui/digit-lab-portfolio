/* TensorFlow.js training in a worker: no server and no student data uploads. */
importScripts('tf.min.js', 'tf-backend-wasm.js');
let dataset, model, modelLayers, running = false;
const progress = state => postMessage({type: 'state', state});
async function loadData() {
  if (dataset) return dataset;
  progress({message: 'Loading browser training engine…'});
  tf.wasm.setWasmPaths(new URL('./', self.location.href).href);
  tf.wasm.setThreadsCount(1);
  await tf.setBackend('wasm');
  await tf.ready();
  progress({message: 'Downloading MNIST digits (about 10 MB, first run only)…'});
  const response = await fetch('mnist-training.bin.gz');
  if (!response.ok) throw Error('Could not load MNIST. Check your connection and try again.');
  if (!self.DecompressionStream) throw Error('Please use a current version of Chrome, Edge, Firefox, or Safari.');
  const bytes = new Uint8Array(await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  if (bytes.length !== 47100000) throw Error('Dataset is incomplete. Reload and try again.');
  const values = Float32Array.from(bytes.subarray(0, 60000 * 784), x => x / 255);
  const labels = bytes.subarray(60000 * 784);
  const xs = tf.tensor2d(values.subarray(0, 50000 * 784), [50000, 784]);
  const ys = tf.tidy(() => tf.oneHot(tf.tensor1d(labels.subarray(0, 50000), 'int32'), 10));
  dataset = {xs, ys, bytes, labels};
  return dataset;
}
function checkLayers(layers) {
  if (!Array.isArray(layers) || layers.length < 1 || layers.length > 6) throw Error('Choose 1–6 hidden layers.');
  for (const l of layers) if (!Number.isInteger(l.neurons) || l.neurons < 1 || l.neurons > 256 ||
    !['relu','sigmoid','tanh'].includes(l.activation)) throw Error('Invalid layer settings.');
}
async function train(layers, epochs) {
  if (running) throw Error('Training is already running.');
  checkLayers(layers);
  if(!Number.isInteger(epochs)||epochs<1||epochs>50)throw Error('Choose 1–50 whole epochs.');
  running = true;
  let candidate;
  progress({status:'training', progress:0, epoch:0, curve:[], message:'Preparing your experiment…'});
  try {
    const data = await loadData();
    candidate = tf.sequential();
    layers.forEach((l, i) => candidate.add(tf.layers.dense({
      ...(i === 0 ? {inputShape:[784]} : {}), units:l.neurons, activation:l.activation,
      kernelInitializer:tf.initializers.glorotUniform({seed:42+i}), biasInitializer:'zeros'
    })));
    candidate.add(tf.layers.dropout({rate:.2, seed:42}));
    candidate.add(tf.layers.dense({units:10, activation:'softmax',
      kernelInitializer:tf.initializers.glorotUniform({seed:42+layers.length})}));
    candidate.compile({optimizer:tf.train.adam(.001), loss:'categoricalCrossentropy', metrics:['accuracy']});
    const curve = [];
    const started = performance.now();
    let round = 0;
    await candidate.fit(data.xs, data.ys, {
      epochs, batchSize:32, shuffle:false, yieldEvery:100,
      callbacks:{
        onEpochBegin:async epoch => {round=epoch; progress({epoch:epoch+1,message:`Training round ${epoch+1} of ${epochs} in your browser…`});},
        onBatchEnd:async batch => {if(batch%50===0) progress({progress:(round+(batch+1)/1563)/epochs});},
        onEpochEnd:async (epoch, logs) => {
          curve.push({epoch:epoch+1,accuracy:logs.acc,loss:logs.loss});
          progress({progress:(epoch+1)/epochs,curve:[...curve]});
        }
      }
    });
    const seconds=(performance.now()-started)/1000;
    progress({message:'Checking 10,000 unseen validation digits…'});
    let correct=0;
    const successes=[], failures=[];
    for(let base=50000;base<60000;base+=256) {
      const count=Math.min(256,60000-base);
      const output=tf.tidy(()=>candidate.predict(tf.tensor2d(
        Float32Array.from(data.bytes.subarray(base*784,(base+count)*784),x=>x/255),[count,784])));
      const scores=await output.data();output.dispose();
      for(let i=0;i<count;i++){
        const slice=scores.subarray(i*10,i*10+10);
        let prediction=0;for(let d=1;d<10;d++)if(slice[d]>slice[prediction])prediction=d;
        const actual=data.labels[base+i],ok=prediction===actual;
        if(ok)correct++;
        const examples=ok?successes:failures;
        if(examples.length<6)examples.push({actual,predicted:prediction,confidence:slice[prediction],
          pixels:Array.from(data.bytes.subarray((base+i)*784,(base+i+1)*784))});
      }
    }
    const result={id:Date.now(),layers,epochs,accuracy:correct/10000,correct,seconds,
      parameters:candidate.countParams(),curve,examples:[...successes,...failures]};
    if(model){model.optimizer.dispose();model.dispose();}
    model=candidate;candidate=null;modelLayers=layers;
    progress({status:'done',progress:1,result,model_ready:true,model_layers:layers,
      message:'Experiment complete. Change one thing and try again.'});
  } catch(e) {
    if(candidate){candidate.optimizer?.dispose();candidate.dispose();}
    progress({status:'error',message:e.message});
  } finally {running=false;}
}
self.onmessage=async ({data})=>{
  try {
    if(data.type==='train'){await train(data.layers,data.epochs);return;}
    if(data.type==='predict'){
      if(running||!model)throw Error('Train a network and wait for it to finish first.');
      if(!Array.isArray(data.pixels)||data.pixels.length!==784 ||
        data.pixels.some(x=>!Number.isFinite(x)||x<0||x>255)||!data.pixels.some(x=>x>0))
        throw Error('Draw a digit first.');
      const output=tf.tidy(()=>model.predict(tf.tensor2d(data.pixels,[1,784]).div(255)));
      const scores=Array.from(await output.data());output.dispose();
      const digit=scores.indexOf(Math.max(...scores));
      postMessage({type:'reply',id:data.id,result:{digit,scores,layers:modelLayers}});
    }
  }catch(e){postMessage({type:'reply',id:data.id,error:e.message});}
};

