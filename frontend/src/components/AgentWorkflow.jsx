function formatToolName(tool) {
  return String(tool || 'workflow_step')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

export default function AgentWorkflow({ trace, model, device, inferenceTime }) {
  const steps = trace?.steps ?? []

  return (
    <section className="panel workflow-panel" aria-labelledby="workflow-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Execution audit</p>
          <h2 id="workflow-title">Agent Workflow</h2>
        </div>
        <span className="status-pill">{trace?.agent || 'BreastCareAgent'}</span>
      </div>

      <dl className="workflow-meta">
        <div><dt>Agent</dt><dd>{trace?.agent || 'BreastCareAgent'}</dd></div>
        <div><dt>Model</dt><dd>{model?.name || 'Not reported'}</dd></div>
        <div><dt>Device</dt><dd>{device || 'Not reported'}</dd></div>
        <div><dt>Inference time</dt><dd>{Number(inferenceTime ?? 0).toFixed(3)} s</dd></div>
      </dl>

      <ol className="workflow-steps">
        {steps.map((step, index) => (
          <li key={`${step.tool}-${index}`}>
            <span className="step-number">{index + 1}</span>
            <span className="step-tool">{formatToolName(step.tool)}</span>
            <span className={`step-status status-${step.status || 'unknown'}`}>
              {step.status || 'unknown'}
            </span>
          </li>
        ))}
      </ol>
    </section>
  )
}
