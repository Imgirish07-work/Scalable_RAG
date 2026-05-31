// Shared attachment configuration — used by Prompt Playground and Model Playground.
// Keep ATTACH_ACCEPT in sync with backend `app/services/file_extractor.py:SUPPORTED_ALL`.
import filePythonIcon   from "../Assets/svg/file-python.svg";
import fileJsIcon       from "../Assets/svg/file-js.svg";
import fileReactIcon    from "../Assets/svg/file-react.svg";
import fileTsIcon       from "../Assets/svg/file-ts.svg";
import fileHtmlIcon     from "../Assets/svg/file-html.svg";
import fileCssIcon      from "../Assets/svg/file-css.svg";
import fileJsonIcon     from "../Assets/svg/file-json.svg";
import fileConfigIcon   from "../Assets/svg/file-config.svg";
import fileTerminalIcon from "../Assets/svg/file-terminal.svg";
import fileTextIcon     from "../Assets/svg/file-text.svg";
import fileJavaIcon     from "../Assets/svg/file-java.svg";
import fileGoIcon       from "../Assets/svg/file-go.svg";
import fileRustIcon     from "../Assets/svg/file-rust.svg";
import fileRubyIcon     from "../Assets/svg/file-ruby.svg";
import filePhpIcon      from "../Assets/svg/file-php.svg";
import fileCsharpIcon   from "../Assets/svg/file-csharp.svg";
import fileCppIcon      from "../Assets/svg/file-cpp.svg";
import fileSwiftIcon    from "../Assets/svg/file-swift.svg";
import fileKotlinIcon   from "../Assets/svg/file-kotlin.svg";
import fileXmlIcon      from "../Assets/svg/file-xml.svg";
import filePdfIcon      from "../Assets/svg/file-pdf.svg";
import fileDocIcon      from "../Assets/svg/file-doc.svg";
import filePptxIcon     from "../Assets/svg/file-pptx.svg";
import fileCsvIcon      from "../Assets/svg/file-csv.svg";
import fileImageIcon    from "../Assets/svg/file-image.svg";

export const ATTACH_ACCEPT =
  ".csv,.xlsx,.xls,.pdf,.docx,.pptx,.txt,.json,.png,.jpg,.jpeg,.gif,.webp," +
  ".py,.java,.js,.ts,.jsx,.tsx,.cpp,.c,.cs,.go,.rs,.php,.rb,.swift,.kt," +
  ".html,.css,.xml,.yaml,.yml,.sh";

export const ATTACH_MAX_MB    = 10;
export const ATTACH_MAX_BYTES = ATTACH_MAX_MB * 1024 * 1024;

const ATTACH_ICONS = {
  csv: fileCsvIcon, xlsx: fileCsvIcon, xls: fileCsvIcon,
  pdf: filePdfIcon,
  docx: fileDocIcon, pptx: filePptxIcon,
  txt: fileTextIcon,
  json: fileJsonIcon,
  png: fileImageIcon, jpg: fileImageIcon, jpeg: fileImageIcon, gif: fileImageIcon, webp: fileImageIcon,
  py: filePythonIcon,
  java: fileJavaIcon,
  js: fileJsIcon, jsx: fileReactIcon,
  ts: fileTsIcon, tsx: fileReactIcon,
  cpp: fileCppIcon, c: fileCppIcon,
  cs: fileCsharpIcon,
  go: fileGoIcon,
  rs: fileRustIcon,
  php: filePhpIcon,
  rb: fileRubyIcon,
  swift: fileSwiftIcon,
  kt: fileKotlinIcon,
  html: fileHtmlIcon,
  css: fileCssIcon,
  xml: fileXmlIcon,
  yaml: fileConfigIcon, yml: fileConfigIcon,
  sh: fileTerminalIcon,
};

export const getAttachIcon = (name) =>
  ATTACH_ICONS[(name || "").split(".").pop().toLowerCase()] || fileTextIcon;
