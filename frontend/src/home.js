import React, { useEffect, useState, useRef } from "react";
import { useTheme, useMediaQuery } from "@mui/material";
import { useNavigate } from "react-router-dom";
import {
	Box,
	Typography,
	TextField,
	Checkbox,
	FormControlLabel,
	FormGroup,
	Button,
	List,
	ListItem,
	ListItemButton,
	ListItemText,
	Divider,
	Paper,
	IconButton,
	Drawer
} from "@mui/material";
import MenuIcon from '@mui/icons-material/Menu';

// Github取得オプション
const GITHUB_OPTIONS = [
	{ key: "issue", label: "Issue" },
	{ key: "pull_request", label: "Pull Request" },
	{ key: "events", label: "Events" },
	{ key: "releases", label: "Releases" }
];

// ダミープロジェクト一覧
const DUMMY_PROJECTS = [
	{ id: 1, name: "Portfolio Generator", description: "ポートフォリオ自動生成ツール" },
	{ id: 2, name: "Quiz App", description: "クイズ学習アプリ" },
	{ id: 3, name: "Blog System", description: "ブログ投稿管理" }
];

const SIDEBAR_WIDTH = 260;
const SIDEBAR_COLLAPSED_WIDTH = 56;

const NEW_PROJECT = { id: 'new', name: '', description: '' };

const HomePage = () => {
	const [apiResult, setApiResult] = useState(null);
	const navigate = useNavigate();
	const [protectedResult, setProtectedResult] = useState(null);
	// 初期選択は新規作成
	const [selectedProject, setSelectedProject] = useState(NEW_PROJECT);
	const [githubUser, setGithubUser] = useState("");
	const [githubRepo, setGithubRepo] = useState("");
	const [selectedOptions, setSelectedOptions] = useState({});
	const [projectName, setProjectName] = useState("");
	const [projectDesc, setProjectDesc] = useState("");
	const [geminiPrompt, setGeminiPrompt] = useState("");
	const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
	const [sidebarHovered, setSidebarHovered] = useState(false);
	const [drawerOpen, setDrawerOpen] = useState(false); // Drawer用
	const sidebarRef = useRef(null);

	// レスポンシブ対応
	const theme = useTheme();
	const isMobile = useMediaQuery(theme.breakpoints.down("sm"));
	const isTablet = useMediaQuery(theme.breakpoints.between("sm", "md"));

	useEffect(() => {
		const token = localStorage.getItem('access_token');
		if (!token) {
			navigate('/');
			return;
		}
		// 認証確認API（既存処理）
		const fetchProtected = async () => {
			try {
				const apiUrl = process.env.REACT_APP_API_URL;
				const apiVersion = process.env.REACT_APP_API_VERSION;
				const url = `${apiUrl}/${apiVersion}/protected`;
				const res = await fetch(url, {
					method: 'GET',
					headers: {
						'Authorization': `Bearer ${token}`
					}
				});
				if (res.ok) {
					const data = await res.json();
					setProtectedResult(data);
				} else {
					localStorage.removeItem('access_token');
					navigate('/');
				}
			} catch (err) {
				setProtectedResult({ error: '通信エラー' });
			}
		};
		fetchProtected();
	}, [navigate]);

	// サイドバーでプロジェクト選択
	const handleProjectSelect = (project) => {
		setSelectedProject(project);
		if (project.id === 'new') {
			setProjectName("");
			setProjectDesc("");
			setGithubUser("");
			setGithubRepo("");
			setSelectedOptions({});
			setGeminiPrompt("");
		} else {
			setProjectName(project.name);
			setProjectDesc(project.description);
		}
	};

	// チェックボックス選択
	const handleCheckboxChange = (key) => {
		setSelectedOptions((prev) => ({
			...prev,
			[key]: !prev[key]
		}));
	};

	// Geminiプロンプト送信
	const handleSubmit = async (e) => {
		e.preventDefault();
		const token = localStorage.getItem('access_token');
		if (!token) {
			alert('認証トークンがありません。再ログインしてください。');
			navigate('/');
			return;
		}
		try {
			const apiUrl = process.env.REACT_APP_API_URL;
			const apiVersion = process.env.REACT_APP_API_VERSION;
			const url = `${apiUrl}/${apiVersion}/project_generate`;
			const res = await fetch(url, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'Authorization': `Bearer ${token}`
				},
				body: JSON.stringify({
					projectName,
					projectDes: projectDesc,
					githubUser,
					githubRepo,
					selectedOptions,
					geminiPrompt
				})
			});
			if (!res.ok) {
				const err = await res.json();
				throw new Error(err.detail || 'APIエラー');
			}
			const data = await res.json();
			setApiResult(data);
		} catch (err) {
			alert("送信失敗: " + err.message);
		}
	};

	// サイドバーのマウスイベント
	const handleSidebarMouseEnter = () => {
		setSidebarHovered(true);
	};
	const handleSidebarMouseLeave = () => {
		setSidebarHovered(false);
	};

	// 実際の表示状態
	// モバイル時はDrawerで管理
	const isSidebarOpen = isMobile ? false : (!sidebarCollapsed || sidebarHovered);

	return (
		<Box
			sx={{
				display: "flex",
				height: "100vh",
				bgcolor: "#f7f8fa",
				flexDirection: isMobile ? "column" : "row"
			}}
		>
				{/* サイドバー（PC/タブレットのみ） */}
				{!isMobile && (
					<Paper
						elevation={2}
						ref={sidebarRef}
						sx={{
							width: isSidebarOpen ? SIDEBAR_WIDTH : SIDEBAR_COLLAPSED_WIDTH,
							minWidth: isSidebarOpen ? SIDEBAR_WIDTH : SIDEBAR_COLLAPSED_WIDTH,
							bgcolor: "#fff",
							borderRadius: 0,
							p: isSidebarOpen ? 2 : 0,
							boxShadow: "0 0 8px #eee",
							transition: "width 0.5s cubic-bezier(.4,0,.2,1)",
							position: "relative",
							zIndex: 2,
							display: "flex",
							flexDirection: "column"
						}}
						onMouseEnter={handleSidebarMouseEnter}
						onMouseLeave={handleSidebarMouseLeave}
					>
						{/* ハンバーガーメニューのみ表示 */}
						{isSidebarOpen ? (
							<Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
								<IconButton
									size="small"
									onClick={() => setSidebarCollapsed((prev) => !prev)}
									aria-label={isSidebarOpen ? "サイドバーを畳む" : "サイドバーを展開"}
									sx={{ mr: 1 }}
								>
									<MenuIcon />
								</IconButton>
							</Box>
						) : (
							<Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", pt: 2 }}>
								<IconButton
									size="small"
									onClick={() => setSidebarCollapsed((prev) => !prev)}
									aria-label={isSidebarOpen ? "サイドバーを畳む" : "サイドバーを展開"}
								>
									<MenuIcon />
								</IconButton>
							</Box>
						)}
						{/* プロジェクトリストは展開時のみ表示（新規作成タブ追加） */}
						{isSidebarOpen && (
							<List>
								<ListItem disablePadding>
									<ListItemButton selected={selectedProject.id === 'new'} onClick={() => handleProjectSelect(NEW_PROJECT)}>
										<ListItemText primary="新規作成" secondary="新しいプロジェクト" />
									</ListItemButton>
								</ListItem>
								{DUMMY_PROJECTS.map((prj) => (
									<ListItem key={prj.id} disablePadding>
										<ListItemButton selected={selectedProject.id === prj.id} onClick={() => handleProjectSelect(prj)}>
											<ListItemText primary={prj.name} secondary={prj.description} />
										</ListItemButton>
									</ListItem>
								))}
							</List>
						)}
					</Paper>
				)}

				{/* モバイル時はヘッダー＋Drawer */}
				{isMobile && (
					<>
						{/* ヘッダー */}
						<Box sx={{ width: "100%", height: 48, bgcolor: "#fff", display: "flex", alignItems: "center", boxShadow: 1, position: "fixed", top: 0, left: 0, zIndex: 100 }}>
							<IconButton
								size="medium"
								onClick={() => setDrawerOpen(true)}
								aria-label="サイドバーを展開"
								sx={{ ml: 1 }}
							>
								<MenuIcon fontSize="medium" />
							</IconButton>
						</Box>
						<Drawer
							anchor="left"
							open={drawerOpen}
							onClose={() => setDrawerOpen(false)}
						>
							<Box sx={{ width: 240, pt: 2 }}>
								<List>
									<ListItem disablePadding>
										<ListItemButton selected={selectedProject.id === 'new'} onClick={() => { handleProjectSelect(NEW_PROJECT); setDrawerOpen(false); }}>
											<ListItemText primary="新規作成" secondary="新しいプロジェクト" />
										</ListItemButton>
									</ListItem>
									{DUMMY_PROJECTS.map((prj) => (
										<ListItem key={prj.id} disablePadding>
											<ListItemButton selected={selectedProject.id === prj.id} onClick={() => { handleProjectSelect(prj); setDrawerOpen(false); }}>
												<ListItemText primary={prj.name} secondary={prj.description} />
											</ListItemButton>
										</ListItem>
									))}
								</List>
							</Box>
						</Drawer>
					</>
				)}

				{/* メイン画面 */}
				<Box
					sx={{
						flex: 1,
						display: "flex",
						justifyContent: "center",
						alignItems: "center",
						width: isMobile ? "100%" : undefined,
						minHeight: isMobile ? 320 : undefined,
						p: isMobile ? 1 : 0,
						position: "relative",
						mt: isMobile ? '48px' : 0 // ヘッダー分の余白
					}}
				>
					<Paper
						elevation={3}
						sx={{
							width: isMobile ? "100%" : isTablet ? 360 : 520,
							p: isMobile ? 2 : 4,
							borderRadius: 4,
							bgcolor: "#fff",
							boxSizing: "border-box"
						}}
					>
						{apiResult ? (
							<Box>
								<Typography variant="h6" sx={{ mb: 2, color: '#333' }}>Gemini出力</Typography>
								{(() => {
									// 1. text部分をパース
									let parsed = null;
									try {
										const text = apiResult.candidates?.[0]?.content?.parts?.[0]?.text;
										parsed = text ? JSON.parse(text) : null;
									} catch (e) {}
									if (!parsed) return <Typography color="error">出力のパースに失敗しました</Typography>;
									return (
										<Box>
											<Typography variant="subtitle1" sx={{ fontWeight: 600 }}>概要</Typography>
											<Typography sx={{ mb: 2 }}>{parsed.projectOverview?.summary}</Typography>
											<Typography variant="subtitle1" sx={{ fontWeight: 600 }}>技術的な工夫点</Typography>
											<ul>
												{parsed.projectOverview?.technicalInnovations?.map((item, i) => (
													<li key={i}>{item}</li>
												))}
											</ul>
											<Typography variant="subtitle1" sx={{ fontWeight: 600 }}>開発ストーリー</Typography>
											<Typography sx={{ mb: 2 }}>{parsed.projectOverview?.developmentStory}</Typography>
											{/* 必要に応じてinterviewAnalysisも同様に展開 */}
										</Box>
									);
								})()}
								<Button variant="outlined" fullWidth sx={{ mt: 2 }} onClick={() => setApiResult(null)}>
									入力フォームに戻る
								</Button>
							</Box>
						) : (
							<form onSubmit={handleSubmit}>
								<Typography variant="h5" sx={{ mb: 2, color: "#333" }}>Github情報</Typography>
								<Typography variant="subtitle1" sx={{ mb: 1 }}>取得するプロジェクト</Typography>
								<Box sx={{ display: "flex", gap: 2, mb: 2 }}>
									<Box sx={{ flex: 1 }}>
										<Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
											<Typography variant="body2" sx={{ fontWeight: 500 }}>
												Githubアカウント名
											</Typography>
											<Typography component="span" sx={{ color: 'error.main', ml: 0.5 }}>*</Typography>
										</Box>
										<TextField
											value={githubUser}
											onChange={e => setGithubUser(e.target.value)}
											fullWidth
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
											value={githubRepo}
											onChange={e => setGithubRepo(e.target.value)}
											fullWidth
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
									value={geminiPrompt}
									onChange={e => setGeminiPrompt(e.target.value)}
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
										value={projectName}
										onChange={e => setProjectName(e.target.value)}
										fullWidth
									/>
								</Box>
								<TextField
									label="プロジェクトの説明(任意)"
									value={projectDesc}
									onChange={e => setProjectDesc(e.target.value)}
									fullWidth
									multiline
									minRows={2}
									sx={{ mb: 2 }}
								/>
								<Button
									variant="contained"
									color="primary"
									type="submit"
									fullWidth
									sx={{ py: 1.2, fontWeight: 600 }}
									disabled={
										!projectName.trim() ||
										!githubUser.trim() ||
										!githubRepo.trim()
									}
								>
									送信
								</Button>
							</form>
						)}
					</Paper>
				</Box>
			</Box>
		);
};

export default HomePage;
