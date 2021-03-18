import { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs'

import './App.css';
function App() {

  const dataToLearn = [
    {
      name: 'Red',
      rgb: [255, 0, 0],
      values: [[255, 160, 122], [250, 128, 114], [233, 150, 122], [240, 128, 128], [205, 92, 92], [220, 20, 60], [178, 34, 34], [255, 0, 0], [139, 0, 0], [128, 0, 0], [255, 99, 71], [255, 69, 0], [219, 112, 147]]
    },
    {
      name: 'Green',
      rgb: [0, 255, 0],
      values: [[124, 252, 0], [127, 255, 0], [50, 205, 50], [34, 139, 34], [0, 128, 0], [0, 100, 0], [173, 255, 47], [154, 205, 50], [0, 255, 127], [0, 250, 154], [144, 238, 144], [152, 251, 152], [143, 188, 143], [60, 179, 113], [32, 178, 170], [46, 139, 87], [128, 128, 0], [85, 107, 47], [107, 142, 35]]
    },
    {
      name: 'Blue',
      rgb: [0, 0, 255],
      values: [[240, 248, 255], [230, 230, 250], [176, 224, 230], [173, 216, 230], [135, 206, 250], [135, 206, 235], [0, 191, 255], [176, 196, 222], [30, 144, 255], [100, 149, 237], [70, 130, 180], [95, 158, 160], [123, 104, 238], [106, 90, 205], [72, 61, 139], [65, 105, 225], [0, 0, 255], [0, 0, 205], [0, 0, 139], [0, 0, 128], [25, 25, 112], [138, 43, 226], [75, 0, 130]]
    },
    {
      name: 'Yellow',
      rgb: [255, 255, 0],
      values: [[255, 255, 204], [255, 255, 153], [255, 255, 102], [255, 255, 51], [255, 255, 0], [204, 204, 0], [153, 153, 0]]
    },
    {
      name: 'Orange',
      rgb: [255, 128, 0],
      values: [[255, 127, 80], [255, 99, 71], [255, 69, 0], [255, 165, 0], [255, 140, 0]]
    },
    {
      name: 'Purple',
      rgb: [128, 0, 128],
      values: [[230, 230, 250], [216, 191, 216], [221, 160, 221], [238, 130, 238], [218, 112, 214], [255, 0, 255], [255, 0, 255], [186, 85, 211], [147, 112, 219], [138, 43, 226], [148, 0, 211], [153, 50, 204], [139, 0, 139], [128, 0, 128], [75, 0, 130]]
    },
    {
      name: 'Pink',
      rgb: [255, 128, 255],
      values: [[255, 192, 203], [255, 182, 193], [255, 105, 180], [255, 20, 147], [219, 112, 147], [199, 21, 133]]
    },
    {
      name: 'Gray',
      rgb: [128, 128, 128],
      values: [[220, 220, 220], [211, 211, 211], [192, 192, 192], [169, 169, 169], [128, 128, 128], [105, 105, 105], [119, 136, 153], [112, 128, 144], [47, 79, 79], [0, 0, 0]]
    },
    {
      name: 'Brown',
      rgb: [128, 0, 0],
      values: [[255, 248, 220], [255, 235, 205], [255, 228, 196], [255, 222, 173], [245, 222, 179], [222, 184, 135], [210, 180, 140], [188, 143, 143], [244, 164, 96], [218, 165, 32], [205, 133, 63], [210, 105, 30], [139, 69, 19], [160, 82, 45], [165, 42, 42], [128, 0, 0]]
    }
  ]

  const [r, setR] = useState(0)
  const [g, setG] = useState(0)
  const [b, setB] = useState(0)


  const [inputsToLearn, setInputsToLearn] = useState([]);
  const [answersToLearn, setAnswersToLearn] = useState([]);

  const [tsInputsToLearn, setTSInputsToLearn] = useState([]);
  const [tsAnswersToLearn, setTSAnswersToLearn] = useState([]);

  const [model, setModel] = useState();

  const [predictedName, setPredictedName] = useState('');
  const [predictedValue, setPredictedValue] = useState([]);

  const [isReady, setIsReady] = useState(true);

  const [correctionShow, setCorrectionShow] = useState(false);

  const [haveModel, setHaveModel] = useState(false);
  const [isTrain, setIsTrain] = useState(false);

  const handleFormatDataToLearn = () => {
    const tempInputs = [];
    const tempAnswers = [];

    dataToLearn.forEach((data) => {
      data.values.forEach((value) => {
        tempInputs.push(value)
        tempAnswers.push(data.rgb)
      })
    })

    setInputsToLearn(tempInputs);
    setAnswersToLearn(tempAnswers);
  }

  const handleCreateModel = () => {
    try {
      //Input formated data to TensorFlow
      //params: (data), [array.length, array properties(0,2,3)]
      setTSInputsToLearn(tf.tensor2d(inputsToLearn, [inputsToLearn.length, 3]))
      setTSAnswersToLearn(tf.tensor2d(answersToLearn, [answersToLearn.length, 3]))

      // Create model
      const tempModel = tf.sequential();

      //Input settings
      tempModel.add(tf.layers.dense({ inputShape: 3, units: 10, useBias: true }));

      //Output settings
      tempModel.add(tf.layers.dense({ inputShape: 10, units: 3, useBias: true }));

      // model compilation
      tempModel.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
      })

      setModel(tempModel)
      setHaveModel(true)
      console.log('Created')
    } catch (error) {
      console.log(error)
    }
  }


  const handleTrainModel = async () => {
    setIsReady(false)
    try {
      for (let i = 0; i < 15; i++) {
        const resTrain = await model.fit(tsInputsToLearn, tsAnswersToLearn, { epochs: 20 }); //epochs = Épocas [Processa 20x para cada trinamento. Quanto mais melhor]
      }
      setIsTrain(true)
    } catch (error) {
      console.log('error train: ', error.message)
    }
    setIsReady(true)
  }


  const handlePredictColor = async (colorRGB) => {
    // get input data to test
    const colorToPredict = tf.tensor2d(colorRGB, [1, 3])

    // Execute 
    const resPredict = model.predict(colorToPredict);

    const predictData = await resPredict.data();

    const maxValue = Math.max(...predictData);

    const index = predictData.indexOf(maxValue);

    setPredictedName(dataToLearn[index].name);
    setPredictedValue(dataToLearn[index].rgb);

    console.log('The color is: ', dataToLearn[index].name, dataToLearn[index].rgb)
    return dataToLearn[index].name;
  }

  const handleSaveModel = async () => {
    setIsReady(false)
    await model.save('localstorage://predict-colors-model');
    setIsReady(true)
  }

  const handleLoadModel = async () => {
    setIsReady(false)
    const tempModel = await tf.loadLayersModel('localstorage://predict-colors-model')
    // model compilation
    tempModel.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    })
    setModel(tempModel);
    setIsTrain(true)
    setIsReady(true)
  }


  const handleRigth = async () => {
    const input = tf.tensor2d([parseInt(r), parseInt(g), parseInt(b)], [1, 3]);
    const answer = tf.tensor2d(predictedValue, [1, 3]);
    await model.fit(input, answer, { epochs: 1 });
  }

  const handleSubmitCorrectColor = async e => {
    const input = tf.tensor2d([parseInt(r), parseInt(g), parseInt(b)], [1, 3]);
    const answer = tf.tensor2d(e.target.value.split`,`.map(x=>+x), [1, 3]);
    await model.fit(input, answer, { epochs: 1 });
    setCorrectionShow(false)
  }

  useEffect(() => {
    handleFormatDataToLearn()
  }, [])
  return (
    <div className="App">
      <div className="sidebarContainer">
      <h2> Neural Network to Predicted Colors </h2>
        <button onClick={handleCreateModel} disabled={haveModel}> Create Model </button>
        <button onClick={handleLoadModel} disabled={haveModel}> Load Model </button>
        <button onClick={handleTrainModel} disabled={isTrain}> Train Model </button>
        <button onClick={handleSaveModel} disabled={!isTrain}> Save Model </button>
      </div>

      <div className="mainContainer">
        <div style={{ margin: 'auto', width: 80, height: 80, border: '1px black solid', backgroundColor: 'rgb(' + r + ',' + g + ',' + b + ')' }} />
        <br />
        <b> RGB({r+','+g+','+b}) </b>
        <div style={{ textAlign: 'center' }}>
          <input disabled={!isTrain} type="range" min="0" max="255" value={r} onChange={e => setR(e.target.value)} />
          <input disabled={!isTrain} type="range" min="0" max="255" value={g} onChange={e => setG(e.target.value)} />
          <input disabled={!isTrain} type="range" min="0" max="255" value={b} onChange={e => setB(e.target.value)} />
        </div>

        <button onClick={() => handlePredictColor([parseInt(r), parseInt(g), parseInt(b)])} disabled={!isTrain}> Predict Color </button>

        <div>
          <h3 style={{ padding: 5, textAlign: 'center' }}>Colos is: {predictedName}</h3>
          <button onClick={handleRigth} disabled={!isTrain && (predictedName.length > 0)} id="btnRigth"> Rigth </button>
          <button onClick={e=>setCorrectionShow(true)} disabled={!isTrain && (predictedName.length > 0)} id="btnWrong"> Wrong </button>
        </div>


      </div>
      {<div className="loading" style={isReady ? { visibility: 'hidden' } : { visibility: 'visible' }}><h1> Loading... </h1></div>}
      {<div className="correction" style={correctionShow ? { visibility: 'visible' } : { visibility: 'hidden' }}>
        <h3> What color is correct? </h3>
        <button className="btnCorrection" value={[255, 0, 0]} onClick={handleSubmitCorrectColor}> Red </button>
        <button className="btnCorrection" value={[0, 255, 0]} onClick={handleSubmitCorrectColor}> Green </button>
        <button className="btnCorrection" value={[0, 0, 255]} onClick={handleSubmitCorrectColor}> Blue </button>
        <button className="btnCorrection" value={[255, 255, 0]} onClick={handleSubmitCorrectColor}> Yellow </button>
        <button className="btnCorrection" value={[255, 128, 0]} onClick={handleSubmitCorrectColor}> Orange </button>
        <button className="btnCorrection" value={[128, 0, 128]} onClick={handleSubmitCorrectColor}> Purple </button>
        <button className="btnCorrection" value={[255, 128, 255]} onClick={handleSubmitCorrectColor}> Pink </button>
        <button className="btnCorrection" value={[128, 128, 128]} onClick={handleSubmitCorrectColor}> Gray </button>
        <button className="btnCorrection" value={[128, 0, 0]} onClick={handleSubmitCorrectColor}> Brown </button>
      </div>}
    </div>
  );
}

export default App;
