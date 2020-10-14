import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import Webcam from 'react-webcam';
import './App.css';


function App() {
  const [img, setImg] = useState(null);
  const [model, setModel] = useState(null);
  const [mobileNet, setMobileNet] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const webcamRef = useRef(null);
  const imgRef = useRef(null);

  useEffect(() => {
      async function run() {
          const m = await tf.loadLayersModel('/model/model.json');
          const mobilenet = await tf.loadLayersModel(
              'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json'
          );
          const layer = mobilenet.getLayer('conv_pw_13_relu');
          setModel(m);
          setMobileNet(
              tf.model({ inputs: mobilenet.inputs, outputs: layer.output })
          );
      }
      run();
  }, []);

  const capture = useCallback(() => {
      setImg(webcamRef.current.getScreenshot());
  }, [webcamRef, setImg]);

  const classify = () =>
      tf.tidy(() => {
          let webcamImage = tf.browser.fromPixels(imgRef.current);

          // Normalize the image between [-1, 1]
          webcamImage = webcamImage.expandDims(0);
          webcamImage = webcamImage
              .toFloat()
              .div(tf.scalar(127))
              .sub(tf.scalar(1));

          const activation = mobileNet.predict(webcamImage);
          let predictions = model.predict(activation);
          setPredictions(predictions.as1D().argMax());
      });

  const videoConstraints = {
      width: 244,
      height: 244,
      facingMode: 'user',
  };

  const handleButtonClick = () => {
      if (!img) {
          capture();
      } else {
          setImg(null);
      }
  };

  
  return (
      <div className='App'>
          <div className='webcam'>
              {img ? (
                  <img ref={imgRef} src={img} alt='Screenshot' />
              ) : (
                  <Webcam
                      audio={false}
                      ref={webcamRef}
                      screenshotFormat='image/jpeg'
                      videoConstraints={videoConstraints}
                  />
              )}
          </div>
          <div>
              <button onClick={handleButtonClick}>
                  {' '}
                  {!img ? 'Capture' : 'Reset'}{' '}
              </button>
              <button onClick={classify} className='classify'>
                  Classify
              </button>
          </div>
          <div>{model && model.summary()}</div>
      </div>
  );
}

export default App;
