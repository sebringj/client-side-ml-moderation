import React, {useState, useEffect, useRef, CSSProperties} from 'react';
import './App.css';
import '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-cpu'
import '@tensorflow/tfjs-backend-webgl'
import * as mobilenet from '@tensorflow-models/mobilenet'
import * as cocoSsd from '@tensorflow-models/coco-ssd'
import * as toxicity from '@tensorflow-models/toxicity'
import * as nsfwjs from 'nsfwjs'

function App() {
  const [imgSrc, setImgSrc] = useState('')
  const [text, setText] = useState('')
  const [classifyImage, setClassifyImage] = useState<{
    className: string;
    probability: number;
  }[]>()
  const [nudifyImage, setNudityImage] = useState<nsfwjs.predictionType[]>()
  const [objectDetection, setObjectDetection] = useState<cocoSsd.DetectedObject[]>()
  const classifyModel = useRef<mobilenet.MobileNet>()
  const objectModel = useRef<cocoSsd.ObjectDetection>()
  const nudityModel = useRef<nsfwjs.NSFWJS>()
  const [classifyText, setClassifyText] = useState<{
    label: string;
    results: {
        probabilities: Float32Array;
        match: boolean;
    }[];
}[]>()
  const toxicityModel = useRef<toxicity.ToxicityClassifier>()
  useEffect(() => {
    async function setModel() {
      const [cm, om, tm, nm] = await Promise.all([
        mobilenet.load(),
        cocoSsd.load(),
        toxicity.load(0.9, [
          'identity_attack',
          'insult',
          'obscene',
          'severe_toxicity',
          'sexual_explicit',
          'threat',
          'toxicity'
        ]),
        nsfwjs.load('/models/nsfw/', { size: 299 })
      ])
      classifyModel.current = cm
      objectModel.current = om
      toxicityModel.current = tm
      nudityModel.current = nm
      console.log('loaded models')
    }
    setModel()
  }, [])
  const checkImage = async () => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    function catchIt(err: Error) {
      console.log(err)
      return undefined
    }
    img.addEventListener('load', async () => {
      await Promise.all([
        objectModel.current?.detect(img).catch(catchIt).then(setObjectDetection),
        classifyModel.current?.classify(img).catch(catchIt).then(setClassifyImage),
        nudityModel.current?.classify(img).catch(catchIt).then(setNudityImage)
      ])
    })
    img.addEventListener('error', (err) => {
      console.log(err)
    })
    img.src = imgSrc
  }
  const checkText = async () => {
    setClassifyText(await toxicityModel.current?.classify(text))
  }
  return (
    <div className="App">
      <label>
        <span>Enter Image Source</span>
        <input type="text" value={imgSrc} onChange={e => setImgSrc(e.currentTarget.value)}  />
      </label>
      <button onClick={checkImage}>Check Image</button>
      {classifyImage && <div>
        <p>Classify Image</p>
        {classifyImage?.map((p, i) => {
          return (
            <div key={`${i}`}>
              <div>{p.className} {p.probability}</div>
            </div>
          )
        })}
      </div>}
      {objectDetection && <div>
        <p>Image Detect Objects</p>
        {objectDetection?.map((p, i) => {
          return (
            <div key={`${i}`}>
              <div>{p.class} {p.score}</div>
            </div>
          )
        })}
      </div>}
      {nudifyImage && <div>
        <p>Image Nudity Detection</p>
        {nudifyImage?.map((p, i) => {
          return (
            <div key={`${i}`}>
              <div>{p.className} {p.probability}</div>
            </div>
          )
        })}
      </div>}
      <div style={styles.textContainer}>
        <label>
          <span>Enter text</span>
          <textarea
            value={text}
            onChange={e => setText(e.currentTarget.value)}
          />
        </label>
      </div>
      <button onClick={checkText}>Check Text</button>
      {classifyText && <table style={styles.classifyTextContainer}>
        <thead>Classify Text</thead>
        {classifyText?.map((p, i) => {
          return (
              <tr key={`${i}`}>{p.results.map((r, j) => {
                return (
                  <td key={`${j}`} style={r.match ? { color: 'red' } : {}}>{p.label}</td>
                )
              })}
            </tr>
          )
        })}
      </table>}
    </div>
  );
}

export default App;

const styles: { [key: string]: CSSProperties } = {
  textContainer: {
    marginTop: 40,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  },
  classifyTextContainer: {
    display: 'table',
    justifySelf: 'center',
    alignItems: 'center'
  }
}