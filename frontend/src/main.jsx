import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import "./index.css";

import Layout from "./components/Layout";
import Home from "./pages/Home";
import Library from "./pages/Library";
import LibraryDetail from "./pages/LibraryDetail";
import Analyze from "./pages/Analyze";
import Gallery from "./pages/Gallery";
import Methodology from "./pages/Methodology";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Home />} />
          <Route path="/library" element={<Library />} />
          <Route path="/library/:className" element={<LibraryDetail />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/analyze/:resultId" element={<Analyze />} />
          <Route path="/gallery" element={<Gallery />} />
          <Route path="/methodology" element={<Methodology />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
