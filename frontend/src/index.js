import React from "react";
import ReactDOM from "react-dom/client";
import LoginRegisterForm from "./login";
import ProjectPage from "./project";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { BrowserRouter, Routes, Route } from "react-router-dom";

const theme = createTheme();
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <BrowserRouter>
    <ThemeProvider theme={theme}>
      <Routes>
        <Route path="/" element={<LoginRegisterForm />} />
        <Route path="/project" element={<ProjectPage />} />
      </Routes>
    </ThemeProvider>
  </BrowserRouter>
);
