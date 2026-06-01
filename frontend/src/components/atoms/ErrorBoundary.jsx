import { Component } from "react";
import { C } from "../../theme";
import Icon from "./Icon";
import warningIcon from "../../Assets/svg/warning.svg";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    this.props.onError?.(error, info);
  }

  reset = () => this.setState({ error: null });

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    if (typeof this.props.fallback === "function") {
      return this.props.fallback({ error, reset: this.reset });
    }
    if (this.props.fallback) return this.props.fallback;

    return (
      <div
        role="alert"
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 12,
          padding: 32,
          color: C.inkSoft,
          textAlign: "center",
        }}
      >
        <Icon src={warningIcon} className="w-7 h-7 text-[var(--c-danger)]" />
        <p style={{ fontSize: 13, margin: 0 }}>Something went wrong.</p>
        <button
          onClick={this.reset}
          style={{
            fontSize: 12,
            padding: "6px 14px",
            borderRadius: 6,
            border: `1px solid ${C.lineSoft}`,
            background: C.bgSoft,
            color: C.ink,
            cursor: "pointer",
          }}
        >
          Try again
        </button>
      </div>
    );
  }
}
