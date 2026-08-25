import { useRef, useState } from 'react'

export default function UploadPanel({ file, previewUrl, onSelect, onAnalyze, loading, error }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  const chooseFile = (candidate) => {
    if (candidate) onSelect(candidate)
  }

  return (
    <section className="panel upload-panel" aria-labelledby="upload-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Step 1</p>
          <h2 id="upload-title">Upload ultrasound image</h2>
        </div>
        <span className="secure-label">Local review workflow</span>
      </div>

      <button
        type="button"
        className={`drop-zone ${dragging ? 'is-dragging' : ''} ${file ? 'has-file' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragEnter={(event) => { event.preventDefault(); setDragging(true) }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault()
          setDragging(false)
          chooseFile(event.dataTransfer.files?.[0])
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/bmp,image/tiff"
          onChange={(event) => chooseFile(event.target.files?.[0])}
          hidden
        />
        {previewUrl ? (
          <img className="upload-preview" src={previewUrl} alt="Selected ultrasound preview" />
        ) : (
          <div className="upload-empty">
            <span className="upload-icon" aria-hidden="true">↑</span>
            <strong>Drop an image here</strong>
            <span>or select a file from your computer</span>
            <small>PNG, JPG, BMP, or TIFF</small>
          </div>
        )}
      </button>

      {file && <p className="file-name">Selected: <strong>{file.name}</strong></p>}
      {error && <p className="error-message" role="alert">{error}</p>}

      <button type="button" className="primary-button" disabled={!file || loading} onClick={onAnalyze}>
        {loading ? <><span className="spinner" /> Analyzing image…</> : 'Analyze image'}
      </button>
    </section>
  )
}
