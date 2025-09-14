import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Box, Tabs, Tab, TextField, Button, Typography } from "@mui/material";

import { useEffect } from "react";

const LoginRegisterForm = () => {
	// ページ表示時にJWTトークンを削除
	// useEffect(() => {
	// 	localStorage.removeItem('access_token');
	// }, []);
	const navigate = useNavigate();
	// 新規登録フォームの状態
	const [registerId, setRegisterId] = useState("");
	const [registerPassword, setRegisterPassword] = useState("");
	const [registerEmail, setRegisterEmail] = useState("");
	const [registerError, setRegisterError] = useState("");

	// ログインフォームの状態
	const [loginId, setLoginId] = useState("");
	const [loginPassword, setLoginPassword] = useState("");
	const [loginError, setLoginError] = useState("");

	// タブの状態（初期値: ログイン）
	const [tab, setTab] = useState(0);
    const API_URL = process.env.REACT_APP_API_URL;
    const API_VERSION = process.env.REACT_APP_API_VERSION;

    const getApiUrl = (endpoint) => {
        return `${API_URL}/${API_VERSION}/${endpoint}`;
    };

	// 新規登録の送信処理
	const handleRegister = async (e) => {
		e.preventDefault();
		setRegisterError("");
		try {
            const url = getApiUrl('register');
            console.log("Register URL:", url); // デバッグ用
			const response = await fetch(url, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					user_id: registerId,
					password: registerPassword,
					email: registerEmail
				})
			});
			if (response.ok) {
				//navigate("/home");
				// 新規登録した場合、そのままhomeに遷移する
				// ただし、JWTトークンを取得してから遷移する
				const data = await response.json();
				if (data.access_token) {
					localStorage.setItem('access_token', data.access_token);
					navigate("/home");
				} else {
					setRegisterError("登録は成功しましたが、認証情報の取得に失敗しました。");
				}
			} else {
				const data = await response.json();
				setRegisterError(data.detail || "登録に失敗しました");
			}
		} catch (error) {
			setRegisterError("通信エラーが発生しました");
		}
	};

	// ログインの送信処理
	const handleLogin = async (e) => {
		e.preventDefault();
		if (!loginId || !loginPassword) {
			setLoginError("ユーザーIDとパスワードは必須です。");
			return;
		}
		setLoginError("");
		try {
			const url = getApiUrl('login');
			const response = await fetch(url, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					user_id: loginId,
					password: loginPassword
				})
			});
			if (response.ok) {
				const data = await response.json();
				// アクセストークンを保存（localStorage）
				localStorage.setItem('access_token', data.access_token);
				navigate("/home");
			} else {
				const data = await response.json();
				setLoginError(data.detail || "ログインに失敗しました");
			}
		} catch (error) {
			setLoginError("通信エラーが発生しました");
		}
	};

	return (
		<Box maxWidth={400} mx="auto" mt={10} p={3} border={1} borderRadius={2} boxShadow={3}>
			<Tabs value={tab} onChange={(e, newValue) => setTab(newValue)} centered>
				<Tab label="ログイン" />
				<Tab label="新規登録" />
			</Tabs>
			{tab === 1 && (
				<Box component="form" onSubmit={handleRegister} mt={2}>
					<TextField
						label="ユーザーID"
						value={registerId}
						onChange={e => setRegisterId(e.target.value)}
						fullWidth
						margin="normal"
						error={!!registerError && !registerId}
					/>
					<TextField
						label="パスワード"
						type="password"
						value={registerPassword}
						onChange={e => setRegisterPassword(e.target.value)}
						//required
						fullWidth
						margin="normal"
						error={!!registerError && !registerPassword}
					/>
					<TextField
						label="メールアドレス（任意）"
						type="email"
						value={registerEmail}
						onChange={e => setRegisterEmail(e.target.value)}
						fullWidth
						margin="normal"
					/>
					{registerError && (
						<Typography color="error" variant="body2" mt={1}>{registerError}</Typography>
					)}
					<Button
						variant="contained"
						color="primary"
						type="submit"
						fullWidth
						sx={{ mt: 2 }}
						disabled={false} // デバッグ用: 常に有効
					>
						新規登録（デバッグモード）
					</Button>
				</Box>
			)}
					{tab === 0 && (
						<>
							<Box component="form" onSubmit={handleLogin} mt={2}>
								<TextField
									label="ユーザーID"
									value={loginId}
									onChange={e => setLoginId(e.target.value)}
									//required
									fullWidth
									margin="normal"
									error={!!loginError && !loginId}
								/>
								<TextField
									label="パスワード"
									type="password"
									value={loginPassword}
									onChange={e => setLoginPassword(e.target.value)}
									//required
									fullWidth
									margin="normal"
									error={!!loginError && !loginPassword}
								/>
								{loginError && (
									<Typography color="error" variant="body2" mt={1}>{loginError}</Typography>
								)}
								{/* パスワードを忘れた場合のワンタイムパスワード発行UI（クリックで表示） */}
								<ForgotPasswordSection />
								<Button variant="contained" color="primary" type="submit" fullWidth sx={{ mt: 2 }}>
									ログイン
								</Button>
							</Box>
						</>
					)}
		</Box>
	);
};

// パスワードを忘れた場合のUI（クリックで表示）
const ForgotPasswordSection = () => {
	const [open, setOpen] = useState(false);
	return (
		<>
			<Typography
				variant="subtitle2"
				gutterBottom
				color="primary"
				sx={{ cursor: "pointer", mt: 2, textDecoration: "underline" }}
				onClick={() => setOpen(!open)}
			>
				パスワードを忘れた場合
			</Typography>
			{open && (
				<Box mt={2} p={2} border={1} borderRadius={2}>
					<Box component="form" onSubmit={e => { e.preventDefault(); alert('ワンタイムパスワードを送信しました'); }}>
						<TextField
							label="登録メールアドレス"
							type="email"
							//required
							fullWidth
							margin="normal"
						/>
						<Button variant="outlined" color="secondary" type="submit" fullWidth sx={{ mt: 1 }}>
							ワンタイムパスワードを送信
						</Button>
					</Box>
				</Box>
			)}
		</>
	);
};

export default LoginRegisterForm;
