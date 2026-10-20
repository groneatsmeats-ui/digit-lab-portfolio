// A local worker replaces the Python API. Nothing is sent to a training server.
const engineState={status:'idle',epoch:0,progress:0,message:'Ready for your first experiment.',result:null,
  history:[],model_ready:false,model_layers:null};
try {
  const saved=JSON.parse(localStorage.getItem('digit-lab-portfolio-history-v1')||'[]');
  engineState.history=saved.filter(r=>Array.isArray(r.layers)&&r.layers.every(l=>
    Number.isInteger(l.neurons)&&l.neurons>=1&&l.neurons<=256&&['relu','sigmoid','tanh'].includes(l.activation))
    &&Number.isFinite(r.accuracy)&&Array.isArray(r.examples)).slice(-30);
}catch{/* Storage may be blocked; experiments still work for this session. */}
const engine=new Worker('trainer.js');
const pending=new Map();let requestId=0;
engine.onmessage=({data})=>{
  if(data.type==='state'){
    Object.assign(engineState,data.state);
    if(data.state.result){
      engineState.history.push(data.state.result);
      engineState.history=engineState.history.slice(-30);
      try{localStorage.setItem('digit-lab-portfolio-history-v1',JSON.stringify(engineState.history));}catch{}
    }
  }else if(data.type==='reply'){
    const request=pending.get(data.id);if(!request)return;
    pending.delete(data.id);clearTimeout(request.timer);
    data.error?request.reject(Error(data.error)):request.resolve(data.result);
  }
};
engine.onerror=()=>{
  engineState.status='error';engineState.model_ready=false;
  engineState.message='Training engine stopped. Reload the page and try a smaller network.';
  for(const request of pending.values()){clearTimeout(request.timer);request.reject(Error(engineState.message));}
  pending.clear();
};
async function localAPI(path,options={}){
  let result;
  if(path==='/api/state')result=engineState;
  else if(path==='/api/train'){
    if(engineState.status==='training')throw Error('A training run is already in progress.');
    const {layers,epochs}=JSON.parse(options.body);
    if(!Number.isInteger(epochs)||epochs<1||epochs>50)throw Error('Choose a whole number of epochs from 1 to 50.');
    engineState.epochs=epochs;engineState.status='training';engineState.progress=0;engineState.epoch=0;engineState.curve=[];
    engineState.message='Starting your experiment…';
    engine.postMessage({type:'train',layers,epochs});result={ok:true};
  }else if(path==='/api/predict'){
    const id=++requestId;
    result=await new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>{pending.delete(id);reject(Error('Prediction timed out. Try again.'));},30000);
      pending.set(id,{resolve,reject,timer});
      engine.postMessage({type:'predict',id,...JSON.parse(options.body)});
    });
  }else throw Error('Unknown local operation.');
  return {ok:true,json:async()=>result};
}
