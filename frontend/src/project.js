import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

const ProjectPage = () => {
	const navigate = useNavigate();
	const [protectedResult, setProtectedResult] = useState(null);
	useEffect(() => {
		const token = localStorage.getItem('access_token');
		if (!token) {
			navigate('/');
			return;
		}
		// テスト用エンドポイントを叩く
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
					setProtectedResult({ error: '認証失敗または権限なし' });
				}
			} catch (err) {
				setProtectedResult({ error: '通信エラー' });
			}
		};
		fetchProtected();
	}, [navigate]);
	return (
		<div style={{ textAlign: "center", marginTop: "100px" }}>
			<h2>プロジェクト画面に遷移しました！</h2>
			<p>ここでプロジェクト一覧や詳細を表示できます。</p>
			<div style={{ marginTop: "40px" }}>
				<h3>テスト用エンドポイントの結果</h3>
				<pre>{protectedResult ? JSON.stringify(protectedResult, null, 2) : '取得中...'}</pre>
			</div>
		</div>
	);
};

export default ProjectPage;
