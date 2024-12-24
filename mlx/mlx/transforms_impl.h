#pragma once

namespace mlx::core::detail {

struct InTracing {
  InTracing() {
    tracing_counter++;
  }
  ~InTracing() {
    tracing_counter--;
  }

  static bool in_tracing() {
    return tracing_counter > 0;
  }

 private:
  static int tracing_counter;
};

struct RetainGraph {
  RetainGraph() {
    tracing_counter++;
  }
  ~RetainGraph() {
    tracing_counter--;
  }

  static bool retain_graph() {
    return tracing_counter > 0;
  }

 private:
  static int tracing_counter;
};

}
