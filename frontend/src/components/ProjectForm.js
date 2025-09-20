import React from "react";
import { Box, TextField, Button, Typography, CircularProgress, FormGroup, FormControlLabel, Checkbox, Divider } from "@mui/material";

const GITHUB_OPTIONS = [
  { key: "issue", label: "Issue" },
  { key: "pull_request", label: "Pull Request" },
  { key: "events", label: "Events" },
  { key: "releases", label: "Releases" }
];

const ProjectForm = ({
  formData,
  handleInputChange,
  handleSubmit,
  isLoading,
  isFormValid,
  requiredFields
}) => {
  // チェックボックスの状態管理
  const [selectedOptions, setSelectedOptions] = React.useState({});

  const handleCheckboxChange = (key) => {
    setSelectedOptions((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // 入力値変更時に親にも反映
  const onInputChange = (e) => {
    handleInputChange(e);
  };


  // 送信時にチェックボックスも含めて送る
  const onSubmit = (e) => {
    e.preventDefault();
    // チェックボックスの値をformDataに追加して送信
    handleSubmit(e, { ...formData, selectedOptions });
  };

  return (
  <Box component="form" onSubmit={onSubmit} sx={{ mt: 2, width: "100%" }}>
      <Typography variant="h5" sx={{ mb: 2, color: "#333" }}>Github情報</Typography>
      <Box sx={{ display: "flex", gap: 2, mb: 2 }}>
        <Box sx={{ flex: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>
              Githubアカウント名
            </Typography>
            <Typography component="span" sx={{ color: 'error.main', ml: 0.5 }}>*</Typography>
          </Box>
          <TextField
            name="githubUser"
            value={formData.githubUser || ""}
            onChange={onInputChange}
            fullWidth
            required
          />
        </Box>
        <Box sx={{ flex: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>
              Githubリポジトリ名
            </Typography>
            <Typography component="span" sx={{ color: 'error.main', ml: 0.5 }}>*</Typography>
          </Box>
          <TextField
            name="githubRepo"
            value={formData.githubRepo || ""}
            onChange={onInputChange}
            fullWidth
            required
          />
        </Box>
      </Box>
      <Typography variant="subtitle1" sx={{ mb: 1 }}>取得したい情報</Typography>
      <FormGroup row sx={{ mb: 2 }}>
        {GITHUB_OPTIONS.map(opt => (
          <FormControlLabel
            key={opt.key}
            control={
              <Checkbox
                checked={!!selectedOptions[opt.key]}
                onChange={() => handleCheckboxChange(opt.key)}
              />
            }
            label={opt.label}
          />
        ))}
      </FormGroup>
      <TextField
        label="Geminiに追加するプロンプト（任意）"
        name="geminiPrompt"
        value={formData.geminiPrompt || ""}
        onChange={onInputChange}
        fullWidth
        multiline
        minRows={2}
        sx={{ mb: 2 }}
      />
      <Divider sx={{ my: 2 }} />
      <Typography variant="h5" sx={{ mb: 2, color: "#333" }}>プロジェクト情報</Typography>
      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
          <Typography variant="body1" sx={{ fontWeight: 500 }}>
            プロジェクト名
          </Typography>
          <Typography component="span" sx={{ color: 'error.main', ml: 0.5 }}>*</Typography>
        </Box>
        <TextField
          name="projectName"
          value={formData.projectName || ""}
          onChange={onInputChange}
          fullWidth
          required
        />
      </Box>
      <TextField
        label="プロジェクトの説明(任意)"
        name="description"
        value={formData.description || ""}
        onChange={onInputChange}
        fullWidth
        multiline
        minRows={2}
        sx={{ mb: 2 }}
        required={false}
      />
      <Box sx={{ mt: 2, display: "flex", justifyContent: "flex-end" }}>
        <Button
          type="submit"
          variant="contained"
          color="primary"
          disabled={!isFormValid || isLoading}
        >
          {isLoading ? <CircularProgress size={24} /> : "送信"}
        </Button>
      </Box>
    </Box>
  );
};

export default ProjectForm;
