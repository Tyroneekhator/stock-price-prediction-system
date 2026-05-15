import { createBrowserRouter } from "react-router";
import { Layout } from "./components/Layout";
import { Dashboard } from "./components/Dashboard";
import { StockDirectory } from "./components/StockDirectory";
import { PredictionWorkspace } from "./components/PredictionWorkspace";
import { PredictionResults } from "./components/PredictionResults";
import { About } from "./components/About";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Layout,
    children: [
      { index: true, Component: Dashboard },
      { path: "directory", Component: StockDirectory },
      { path: "workspace", Component: PredictionWorkspace },
      { path: "results", Component: PredictionResults },
      { path: "about", Component: About },
    ],
  },
]);
