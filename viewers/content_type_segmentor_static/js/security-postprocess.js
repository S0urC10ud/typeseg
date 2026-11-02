// Security's canonical-label decoder: confidence -> visible whitespace -> short runs.
// Browser input is normalized ASCII, so these offsets are also byte offsets.
function securityPostprocessTraced(text, labels, probabilities, options = {}) {
  const count = labels.length;
  const classes = count ? probabilities.length / count : 0;
  const pp = { threshold: true, whitespace: true, shortRuns: true, ...options.ppOptions };
  const stages = [];
  let current = new Int32Array(labels);
  if (pp.threshold) {
    const threshold = options.threshold ?? 0.3;
    for (let i = 0; i < count; i++) {
      let maximum = 0;
      for (let c = 0; c < classes; c++) maximum = Math.max(maximum, probabilities[i * classes + c]);
      if (maximum < threshold) current[i] = options.otherId ?? classes;
    }
    stages.push({ step: 'threshold', labels: new Int32Array(current) });
  }
  if (pp.whitespace) {
    const visible = ch => !' \t\r\n'.includes(ch);
    const left = new Int32Array(count).fill(-1), right = new Int32Array(count).fill(-1);
    const localLeft = new Int32Array(count).fill(-1), localRight = new Int32Array(count).fill(-1);
    let previous = -1;
    for (let i = 0; i < count; i++) { if (visible(text[i])) previous = i; left[i] = previous; }
    previous = -1;
    for (let i = count - 1; i >= 0; i--) { if (visible(text[i])) previous = i; right[i] = previous; }
    const starts = [0];
    for (let i = 0; i + 1 < count; i++) if (text[i] === '\n') starts.push(i + 1);
    for (let line = 0; line < starts.length; line++) {
      const start = starts[line], end = starts[line + 1] ?? count;
      previous = -1;
      for (let i = start; i < end; i++) { if (visible(text[i])) previous = i; localLeft[i] = previous; }
      previous = -1;
      for (let i = end - 1; i >= start; i--) { if (visible(text[i])) previous = i; localRight[i] = previous; }
    }
    const original = current;
    current = new Int32Array(current);
    for (let i = 0; i < count; i++) if (!visible(text[i])) {
      let a = localLeft[i], b = localRight[i];
      if (a < 0 && b < 0) { a = left[i]; b = right[i]; }
      const source = a < 0 ? b : b < 0 ? a : i - a <= b - i ? a : b;
      if (source >= 0) current[i] = original[source];
    }
    stages.push({ step: 'whitespace', labels: new Int32Array(current) });
  }
  if (pp.shortRuns) {
    const minimum = options.min_run_chars ?? 3;
    while (minimum > 1) {
      const runs = [];
      let start = 0;
      for (let i = 1; i <= count; i++) if (i === count || current[i] !== current[start]) {
        runs.push({ start, end: i, label: current[start] }); start = i;
      }
      let changed = false;
      for (let i = 1; i + 1 < runs.length; i++) {
        const run = runs[i], left = runs[i - 1], right = runs[i + 1];
        if (run.end - run.start >= minimum) continue;
        const replacement = left.label === right.label || left.end - left.start >= right.end - right.start ? left.label : right.label;
        current.fill(replacement, run.start, run.end); changed = true; break;
      }
      if (!changed) break;
    }
    stages.push({ step: 'shortRuns', labels: new Int32Array(current) });
  }
  return { labels: current, stages };
}
if (typeof module !== 'undefined' && module.exports) module.exports = { securityPostprocessTraced };
