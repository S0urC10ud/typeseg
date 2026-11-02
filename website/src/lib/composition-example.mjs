// Shortened, annotated excerpt. Offsets refer to the displayed excerpt only.
export const sampleSha = '626f2311154e7a07136f4d8d45fa218a359874d0d805c22c52544385bbd1290d';
const types = {
  Batch: { color: '#3976c5', detail: 'Batch assignment syntax and delayed variable expansion.' },
  PowerShell: { color: '#8655c9', detail: 'A piece of PowerShell stored inside a batch variable.' },
  Base64: { color: '#b84f87', detail: 'A fragment of the encoded key within the PowerShell string.' },
  Omitted: { color: '#7d8896', detail: 'An ellipsis marks source omitted for display; this is not a content type.' },
};
const chunks = [
  ['Batch', 'set "VmxRfZ='],
  ['PowerShell', '::PKCS7; $nARvf.Key=[Syst'],
  ['Batch', '"\nset "VzgZGP='],
  ['PowerShell', "em.Convert]::FromBase64String('"],
  ['Base64', '3lg'],
  ['Batch', '"\nset "VqSyhZ='],
  ['Base64', 'PRfefMk'],
  ['Omitted', '...'],
  ['Base64', 'JaLNmV'],
  ['Batch', '"\nset "VnPUJq='],
  ['Base64', 'VBqbwycQkIDPxsvE='],
  ['PowerShell', "'); $nAR"],
  ['Batch', '"\n'],
  ['Omitted', '...'],
  ['Batch', '\nset "CkBNy='],
  ['Omitted', '...'],
  ['Batch', '!VmxRfZ!!VzgZGP!!VqSyhZ!!VnPUJq!'],
  ['Omitted', '...'],
  ['Batch', '"\n'],
].map(([label, text]) => ({ ...types[label], label, text }));

let offset = 0;
export const regions = chunks.map(chunk => {
  const start = offset;
  offset += new TextEncoder().encode(chunk.text).length;
  return { ...chunk, start, end: offset };
});
export const sampleText = chunks.map(chunk => chunk.text).join('');
export const totalBytes = offset;

/**
 * @param {string} text
 * @param {{ start: number, end: number, color: string }[]} annotations
 * @returns {{ number: number, parts: { regionIndex: number, text: string, color: string }[] }[]}
 */
export function buildSourceLines(text, annotations) {
  let start = 0;
  return text.split('\n').map((line, index) => {
    const end = start + line.length;
    const parts = annotations.flatMap((region, regionIndex) => {
      const from = Math.max(start, region.start);
      const to = Math.min(end, region.end);
      return to > from ? [{ regionIndex, text: text.slice(from, to), color: region.color }] : [];
    });
    start = end + 1;
    return { number: index + 1, parts };
  });
}
