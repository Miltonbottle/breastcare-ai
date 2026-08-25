export default function AssistedReview({ analysis }) {
  return (
    <section className="panel review-panel" aria-labelledby="review-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Structured output</p>
          <h2 id="review-title">AI-Assisted Review</h2>
        </div>
      </div>

      <div className="review-summary">
        <div><span>Segmented extent</span><strong>{analysis.lesion_extent}</strong></div>
        <div><span>Geometric description</span><strong>{analysis.shape_description}</strong></div>
      </div>

      <h3>Review flags</h3>
      <ul className="review-list">
        {(analysis.review_flags ?? []).map((flag) => <li key={flag}>{flag}</li>)}
      </ul>

      <div className="limitations-box">
        <h3>Limitations and disclaimer</h3>
        <ul>
          {(analysis.limitations ?? []).map((item) => <li key={item}>{item}</li>)}
        </ul>
        <p>{analysis.disclaimer}</p>
      </div>
    </section>
  )
}
