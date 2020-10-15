import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import Webcam from 'react-webcam';
import './App.css';

function App() {
    const [img, setImg] = useState(null);
    const [model, setModel] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const webcamRef = useRef(null);
    const imgRef = useRef(null);

    useEffect(() => {
        async function run() {
            const m = await tf.loadLayersModel('/model/model.json');
            setModel(m);
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
            let p = model.predict(webcamImage);
            p = p.as1D().argMax();
            p.data().then((data) => setPrediction(data));
        });

    const handleButtonClick = () => {
        if (!img) {
            capture();
        } else {
            setImg(null);
            setPrediction(null);;
        }
    };

    return (
        <div className='App'>
        <h1>Rock, Paper, Scissors</h1>
            <div className='webcam'>
                {img ? (
                    <img ref={imgRef} src={img} alt='Screenshot' />
                ) : (
                    <Webcam
                        audio={false}
                        ref={webcamRef}
                        screenshotFormat='image/jpeg'
                        videoConstraints={{
                                width: 300,
                                height: 300,
                                facingMode: 'user',
                        }}
                    />
                )}
            </div>
            <div>
                <button onClick={handleButtonClick}>
                    {' '}
                    {!img ? 'Capture' : 'Reset'}{' '}
                </button>
                <button
                    onClick={classify}
                    className='classify'
                    disabled={!img && true}
                >
                    Classify
                </button>
            </div>
            {prediction && (
                <div className='prediction'>
                    {prediction[0] === 0
                        ? 'Rock'
                        : prediction[0] === 1
                        ? 'Paper'
                        : prediction[0] === 2
                        ? 'Scissors'
                        : ''}
                </div>
            )}
        </div>
    );
}

export default App;
