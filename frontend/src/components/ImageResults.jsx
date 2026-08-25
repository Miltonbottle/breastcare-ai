const imageCards = [
  ['Original ultrasound', 'original_path'],
  ['Predicted segmentation overlay', 'overlay_path'],
  ['Binary segmentation mask', 'mask_path'],
]

export default function ImageResults({ image, segmentation }) {
  return (
    <section className="panel" aria-labelledby="imaging-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Imaging output</p>
          <h2 id="imaging-title">Segmentation review</h2>
        </div>
        <span className="status-pill">Analysis complete</span>
      </div>
      <div className="image-grid">
        {imageCards.map(([title, key]) => {
          const source = key === 'original_path' ? image[key] : segmentation[key]
          return (
            <figure className="image-card" key={key}>
              <img src={source} alt={title} />
              <figcaption>{title}</figcaption>
            </figure>
          )
        })}
      </div>
    </section>
  )
}
