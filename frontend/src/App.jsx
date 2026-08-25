import { useEffect, useState } from 'react'
import AgentWorkflow from './components/AgentWorkflow'
import AssistedReview from './components/AssistedReview'
import ImageResults from './components/ImageResults'
import Measurements from './components/Measurements'
import UploadPanel from './components/UploadPanel'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

function toAssetUrl(relativePath) {
  return relativePath?.startsWith('http') ? relativePath : `${API_BASE_URL}${relativePath}`
}

function getApiErrorMessage(payload, status) {
  if (status === 400 || status === 422) {
    return 'Unable to analyze this image. Please upload a breast ultrasound image.'
  }

  if (status === 503) {
    return 'Backend unavailable. Confirm that the local backend is running.'
  }

  if (typeof payload?.detail === 'string') return payload.detail
  if (typeof payload?.detail?.message === 'string') return payload.detail.message
  return 'The image could not be analyzed. Please try another image.'
}

export default function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
  }, [previewUrl])

  const handleSelect = (selectedFile) => {
    if (!selectedFile?.type.startsWith('image/')) {
      setError('Please choose a valid image file.')
      return
    }
    setError('')
    setResult(null)
    setFile(selectedFile)
    setPreviewUrl(URL.createObjectURL(selectedFile))
  }

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    try {
      const formData = new FormData()
      formData.append('image', file)
      const response = await fetch(`${API_BASE_URL}/analyze`, { method: 'POST', body: formData })
      const payload = await response.json().catch(() => ({}))
      if (!response.ok || !payload.success) {
        throw new Error(getApiErrorMessage(payload, response.status))
      }
      const healthPayload = await fetch(`${API_BASE_URL}/health`)
        .then((healthResponse) => healthResponse.ok ? healthResponse.json() : {})
        .catch(() => ({}))
      setResult({
        ...payload,
        device: healthPayload.device,
        image: { ...payload.image, original_path: toAssetUrl(payload.image.original_path) },
        segmentation: {
          ...payload.segmentation,
          mask_path: toAssetUrl(payload.segmentation.mask_path),
          overlay_path: toAssetUrl(payload.segmentation.overlay_path),
        },
      })
    } catch (requestError) {
      const unavailable = requestError instanceof TypeError
      setError(unavailable ? 'Backend unavailable. Confirm that the local backend is running.' : requestError.message)
    } finally {
      setLoading(false)
    }
  }

  const startNewAnalysis = () => {
    setFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError('')
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true">BC</div>
        <div>
          <h1>BreastCare AI</h1>
          <p>AI-assisted breast ultrasound lesion review</p>
        </div>
        <span className="topbar-note">Segmentation-supported workflow</span>
      </header>

      <div className="content-wrap">
        {!result ? (
          <div className="upload-layout">
            <div className="intro-copy">
              <p className="eyebrow">Breast ultrasound workflow</p>
              <h2>Clear, structured segmentation review.</h2>
              <p>Upload an ultrasound image to generate a segmentation overlay and objective geometric measurements for clinician review.</p>
            </div>
            <UploadPanel {...{ file, previewUrl, onSelect: handleSelect, onAnalyze: handleAnalyze, loading, error }} />
          </div>
        ) : (
          <>
            <div className="result-header">
              <div>
                <p className="eyebrow">Completed analysis</p>
                <h2>{result.image.filename}</h2>
                <p>Inference completed in {Number(result.performance.inference_time_seconds).toFixed(3)} seconds.</p>
              </div>
              <button type="button" className="secondary-button" onClick={startNewAnalysis}>New Analysis</button>
            </div>
            <ImageResults image={result.image} segmentation={result.segmentation} />
            <div className="details-grid">
              <Measurements features={result.features} />
              <AssistedReview analysis={result.analysis} />
            </div>
            <AgentWorkflow
              trace={result.agent_trace}
              model={result.model}
              device={result.device}
              inferenceTime={result.performance.inference_time_seconds}
            />
          </>
        )}
      </div>
    </main>
  )
}
