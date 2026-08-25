const formatNumber = (value, digits = 2) => Number(value ?? 0).toFixed(digits)

export default function Measurements({ features }) {
  const measurements = [
    ['Lesion area', `${formatNumber(features.lesion_area_percentage)}%`],
    ['Bounding box width', `${features.bounding_box_width ?? 0} px`],
    ['Bounding box height', `${features.bounding_box_height ?? 0} px`],
    ['Aspect ratio', formatNumber(features.aspect_ratio)],
    ['Circularity', formatNumber(features.circularity)],
    ['Connected components', features.connected_components ?? 0],
  ]

  return (
    <section className="panel" aria-labelledby="measurements-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Geometry</p>
          <h2 id="measurements-title">Lesion Measurements</h2>
        </div>
      </div>
      <dl className="measurement-grid">
        {measurements.map(([label, value]) => (
          <div className="measurement" key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}
