import React from "react";
import ReactDOM from "react-dom/client";
import LoginRegisterForm from "./login";
import ProjectPage from "./project";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from './contexts/AuthContext';
import PrivateRoute from './components/PrivateRoute';

const theme = createTheme();
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <AuthProvider>
    <BrowserRouter>
      <ThemeProvider theme={theme}>
        <Routes>
          {/* ログイン画面のルート */}
          <Route path="/" element={<LoginRegisterForm />} />
          {/* JWT認証が必要な画面のルート */}
          <Route element={<PrivateRoute />}>
              <Route path="/project" element={<ProjectPage />} />
            </Route>
        </Routes>
      </ThemeProvider>
    </BrowserRouter>
  </AuthProvider>
);
