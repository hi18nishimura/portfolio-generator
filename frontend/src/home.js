import React, { useEffect, useState, useRef } from "react";
import { useTheme, useMediaQuery, Box, Paper, IconButton, Drawer, Typography, Button } from "@mui/material";
import { useNavigate } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import ProjectForm from "./components/ProjectForm";
import GeminiResult from "./components/GeminiResult";
import MenuIcon from '@mui/icons-material/Menu';

const NEW_PROJECT = { id: 'new', name: '', description: '' };


const HomePage = () => {
	const [apiResult, setApiResult] = useState(null);
	const navigate = useNavigate();
	const [protectedResult, setProtectedResult] = useState(null);
	const [selectedProject, setSelectedProject] = useState(NEW_PROJECT);
	const [formData, setFormData] = useState({
		projectName: "",
		githubUrl: "",
		description: "",
		additionalInfo: ""
	});
	const [isLoading, setIsLoading] = useState(false);
	const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
	const [sidebarHovered, setSidebarHovered] = useState(false);
	const [drawerOpen, setDrawerOpen] = useState(false);
	const [PROJECT_LIST, setProjectList] = useState([]);
	const sidebarRef = useRef(null);

	const theme = useTheme();
	const isMobile = useMediaQuery(theme.breakpoints.down("sm"));
	const isTablet = useMediaQuery(theme.breakpoints.between("sm", "md"));
	// プロジェクト一覧を取得
	useEffect(() => {
		const fetchProjects = async () => {
			try {
				const token = localStorage.getItem('access_token');
				const apiUrl = process.env.REACT_APP_API_URL;
				const apiVersion = process.env.REACT_APP_API_VERSION;
				const url = `${apiUrl}/${apiVersion}/load_projects`;
				const res = await fetch(url, {
					method: 'GET',
					headers: {
						'Authorization': `Bearer ${token}`
					}
				});
				if (res.ok) {
					const data = await res.json();
					// projects配列からprj_name, prj_descriptionを抽出
					const projects = (data.projects || []).map((prj, idx) => ({
						id: prj.prj_id || idx,
						name: prj.prj_name || '',
						description: prj.prj_description || '',
						prompt: prj.prompt || '',
						output: prj.output || null
					}));
					setProjectList(projects);
				}
			} catch (err) {
				// エラー時は空配列のまま
			}
		};
		fetchProjects();
	}, []);

	useEffect(() => {
		const token = localStorage.getItem('access_token');
		if (!token) {
			navigate('/');
			return;
		}
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
			setFormData({ projectName: "", githubUrl: "", description: "", additionalInfo: "" });
			setApiResult(null);
		} else {
			setFormData({
				projectName: project.name,
				githubUrl: "",
				description: project.description,
				additionalInfo: ""
			});
			// PROJECT_LISTから該当プロジェクトのpromptを取得し、apiResultにセット
			const found = PROJECT_LIST.find(p => p.id === project.id);
			if (found && found.output) {
				setApiResult(found.output);
			} else {
				setApiResult(null);
			}
		}
	};

	// 入力フォーム変更
	const handleInputChange = (e) => {
		const { name, value } = e.target;
		setFormData((prev) => ({ ...prev, [name]: value }));
	};

	// バリデーション
	const requiredFields = ["projectName"];
	const isFormValid = React.useMemo(() => {
		return requiredFields.every((key) => formData[key]?.trim());
	}, [formData, requiredFields]);

	// Geminiプロンプト送信
	const handleSubmit = async (e, submitData) => {
		e.preventDefault();
		const raw = submitData || formData;
		// Model.pyのGeminiPromptRequestに合わせてキー名・構造を修正
		const payload = {
			projectName: raw.projectName || "",
			projectDes: raw.description || "",
			githubUser: raw.githubUser || "",
			githubRepo: raw.githubRepo || "",
			selectedOptions: raw.selectedOptions || {},
			geminiPrompt: raw.geminiPrompt || ""
		};
		const token = localStorage.getItem('access_token');
		if (!token) {
			alert('認証トークンがありません。再ログインしてください。');
			navigate('/');
			return;
		}
		setIsLoading(true);
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
				body: JSON.stringify(payload)
			});
			if (!res.ok) {
				const err = await res.json();
				throw new Error(err.detail || 'APIエラー');
			}
			const data = await res.json();
			setApiResult(data);
		} catch (err) {
			alert("送信失敗: " + err.message);
		} finally {
			setIsLoading(false);
		}
	};

	// サイドバーのマウスイベント
	const handleSidebarMouseEnter = () => setSidebarHovered(true);
	const handleSidebarMouseLeave = () => setSidebarHovered(false);
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
				<Box sx={{
					height: '100vh',
					overflow: 'auto',
					width: isSidebarOpen ? 260 : 56,
					minWidth: isSidebarOpen ? 260 : 56,
					transition: 'width 0.5s cubic-bezier(.4,0,.2,1)',
					bgcolor: '#fff',
					boxShadow: 1,
					position: 'fixed',
					top: 0,
					left: 0,
					zIndex: 1200
				}}>
					<Sidebar
						isSidebarOpen={isSidebarOpen}
						sidebarRef={sidebarRef}
						sidebarCollapsed={sidebarCollapsed}
						sidebarHovered={sidebarHovered}
						handleSidebarMouseEnter={handleSidebarMouseEnter}
						handleSidebarMouseLeave={handleSidebarMouseLeave}
						setSidebarCollapsed={setSidebarCollapsed}
						selectedProject={selectedProject}
						handleProjectSelect={handleProjectSelect}
						PROJECT_LIST={PROJECT_LIST}
						NEW_PROJECT={NEW_PROJECT}
						isMobile={isMobile}
					/>
				</Box>
			)}

			{/* モバイル時はヘッダー＋Drawer */}
			{isMobile && (
				<>
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
							<Sidebar
								isSidebarOpen={true}
								sidebarRef={sidebarRef}
								sidebarCollapsed={false}
								sidebarHovered={false}
								handleSidebarMouseEnter={() => { }}
								handleSidebarMouseLeave={() => { }}
								setSidebarCollapsed={() => { }}
								selectedProject={selectedProject}
								handleProjectSelect={(prj) => { handleProjectSelect(prj); setDrawerOpen(false); }}
								PROJECT_LIST={PROJECT_LIST}
								NEW_PROJECT={NEW_PROJECT}
								isMobile={false}
							/>
						</Box>
					</Drawer>
				</>
			)}

			{/* メイン画面 */}
			<Box
				sx={{
					flex: 1,
					bgcolor: '#f7f8fa',
					p: isMobile ? 1 : 4,
					mt: isMobile ? '48px' : 0,
					ml: !isMobile ? (isSidebarOpen ? '260px' : '56px') : 0,
					transition: 'margin-left 0.5s cubic-bezier(.4,0,.2,1)'
				}}
			>
				<Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start', minHeight: isMobile ? 320 : undefined, width: '100%', height: '100%' }}>
					{apiResult ? (
						<Box sx={{ width: '100%', height: '100%', p: isMobile ? 1 : 4 }}>
							<GeminiResult result={apiResult} />
						</Box>
					) : (
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
							<ProjectForm
								formData={formData}
								handleInputChange={handleInputChange}
								handleSubmit={handleSubmit}
								isLoading={isLoading}
								isFormValid={isFormValid}
								requiredFields={requiredFields}
							/>
						</Paper>
					)}
				</Box>
			</Box>
		</Box>
	);
};

export default HomePage;
