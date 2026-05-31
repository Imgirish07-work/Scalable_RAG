// Lazy-load pdf.js from CDN so it stays out of the main bundle.
// Single shared loader — both FilePreview (modal viewer) and DocPreviewCard
// (thumbnail rendering) call this. Subsequent calls return the cached module.

const PDF_VER = '3.11.174'
const PDF_JS = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${PDF_VER}/pdf.min.js`
const PDF_WK = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${PDF_VER}/pdf.worker.min.js`

let _loadPromise = null

export function loadPdfJs() {
  if (window.pdfjsLib) return Promise.resolve(window.pdfjsLib)
  if (_loadPromise) return _loadPromise
  _loadPromise = new Promise((resolve, reject) => {
    const s = document.createElement('script')
    s.src = PDF_JS
    s.onload = () => {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc = PDF_WK
      resolve(window.pdfjsLib)
    }
    s.onerror = () => {
      _loadPromise = null
      reject(new Error('Failed to load pdf.js'))
    }
    document.head.appendChild(s)
  })
  return _loadPromise
}
