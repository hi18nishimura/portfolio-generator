import React from "react";
import { Box, Typography, Paper, TextField, List, ListItem, ListItemText, Accordion, AccordionSummary, AccordionDetails, Card, CardContent, Stack, Avatar, Chip, Divider } from "@mui/material";
import DescriptionIcon from '@mui/icons-material/Description';
import EmojiObjectsIcon from '@mui/icons-material/EmojiObjects';
import BuildIcon from '@mui/icons-material/Build';
import SchoolIcon from '@mui/icons-material/School';
import QuestionAnswerIcon from '@mui/icons-material/QuestionAnswer';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import Timeline from '@mui/lab/Timeline';
import TimelineItem from '@mui/lab/TimelineItem';
import TimelineSeparator from '@mui/lab/TimelineSeparator';
import TimelineDot from '@mui/lab/TimelineDot';
import TimelineConnector from '@mui/lab/TimelineConnector';
import TimelineContent from '@mui/lab/TimelineContent';


const GeminiResult = ({ result }) => {
  if (!result) return null;

  // Gemini APIの出力からtext部分を抽出しJSONパース
  let parsed = null;
  try {
    // result.candidates[0].content.parts[0].text をパース
    const text = result?.candidates?.[0]?.content?.parts?.[0]?.text;
    if (text) {
      parsed = JSON.parse(text);
    }
  } catch (e) {
    // パース失敗時はnullのまま
  }

  // 各項目ごとに分割表示
  if (parsed) {
    const projectOverview = parsed.projectOverview || {};
    const interviewAnalysis = parsed.interviewAnalysis || {};
    const foundationalKnowledge = interviewAnalysis.foundationalKnowledge || {};
    const expectedQuestions = interviewAnalysis.expectedQuestions || [];

    // タイムライン用：開発ストーリーを文ごとに分割（「。）」「）」で区切らないよう工夫）
    const devStoryLines = (projectOverview.developmentStory || '')
      // 句点の直後にカッコやバッククォート、改行が続く場合はsplitしない
      .replace(/([。．])([）)】】』』】】\`"]|\s*\n)/g, '$1$2')
      // 句点の直後に何もなければ改行を挿入
      .replace(/([。．])(?=[^）)】】』』】】\`"\s\n])/g, '$1\n')
      .split(/\n+/)
      .map(s => s.trim())
      .filter(Boolean);

    return (
      <Box sx={{
        width: '100%',
        maxWidth: 900,
        minWidth: 320,
        mx: 'auto',
        p: { xs: 1, sm: 4 },
        bgcolor: '#fff',
        boxShadow: 1,
        borderRadius: 2,
      }}>
        {/* 概要カード */}
        <Card sx={{ mb: 3, bgcolor: '#e3f2fd' }}>
          <CardContent>
            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
              <Avatar sx={{ bgcolor: '#1976d2', width: 32, height: 32 }}>
                <DescriptionIcon />
              </Avatar>
              <Typography variant="h6" sx={{ fontWeight: 'bold' }}>プロジェクト概要</Typography>
            </Stack>
            <Typography sx={{ whiteSpace: 'pre-line', fontSize: 17 }}>{projectOverview.summary}</Typography>
          </CardContent>
        </Card>

        {/* 技術的な工夫点カード（概要と同じ形式） */}
        <Card sx={{ mb: 3, bgcolor: '#fffde7' }}>
          <CardContent>
            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
              <Avatar sx={{ bgcolor: '#fbc02d', width: 32, height: 32 }}>
                <EmojiObjectsIcon />
              </Avatar>
              <Typography variant="h6" sx={{ fontWeight: 'bold' }}>技術的な工夫点</Typography>
            </Stack>
            <Typography sx={{ whiteSpace: 'pre-line', fontSize: 17 }}>
              {Array.isArray(projectOverview.technicalInnovations)
                ? projectOverview.technicalInnovations.join('\n')
                : projectOverview.technicalInnovations}
            </Typography>
          </CardContent>
        </Card>

        {/* 開発ストーリー：タイムライン風 */}
        <Card sx={{ mb: 3, bgcolor: '#f3e5f5' }}>
          <CardContent>
            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
              <Avatar sx={{ bgcolor: '#8e24aa', width: 32, height: 32 }}>
                <BuildIcon />
              </Avatar>
              <Typography variant="h6" sx={{ fontWeight: 'bold' }}>開発ストーリー</Typography>
            </Stack>
            <Box sx={{ display: 'flex', width: '100%' }}>
              {/* 左カラム：タイムラインUI */}
              <Box sx={{ minWidth: 40, mr: 2, pt: 0.2 }}>
                {devStoryLines.map((_, idx) => (
                  <Box key={idx} sx={{ height: 'auto', minHeight: 32, display: 'flex', alignItems: 'center' }}>
                    <Timeline sx={{ p: 0, m: 0 }}>
                      <TimelineItem sx={{ minHeight: 32, py: 0 }}>
                        <TimelineSeparator>
                          <TimelineDot color="secondary" />
                          {idx < devStoryLines.length - 1 && <TimelineConnector sx={{ minHeight: 32 }} />}
                        </TimelineSeparator>
                      </TimelineItem>
                    </Timeline>
                  </Box>
                ))}
              </Box>
              {/* 右カラム：テキスト */}
              <Box sx={{ flex: 1 }}>
                {devStoryLines.map((line, idx) => (
                  <Box key={idx} sx={{ py: 1.2, fontSize: 15, textAlign: 'left', minHeight: 32, display: 'flex', alignItems: 'center' }}>{line}</Box>
                ))}
              </Box>
            </Box>
          </CardContent>
        </Card>

        {/* 面接対策: 基礎知識 */}
        <Card sx={{ mb: 3, bgcolor: '#e8f5e9' }}>
          <CardContent>
            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
              <Avatar sx={{ bgcolor: '#388e3c', width: 32, height: 32 }}>
                <SchoolIcon />
              </Avatar>
              <Typography variant="h6" sx={{ fontWeight: 'bold' }}>面接対策: 基礎知識</Typography>
            </Stack>
            <Stack spacing={1} sx={{ mb: 2 }}>
              {(foundationalKnowledge.knowledge || []).map((item, idx) => (
                <Paper key={idx} sx={{ p: 1.2, bgcolor: '#f1f8e9' }}>
                  <b>{item.area}</b>: {item.justification}
                </Paper>
              ))}
            </Stack>
            <Divider sx={{ my: 1 }} />
            <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1 }}>説明戦略</Typography>
            <Typography sx={{ whiteSpace: 'pre-line' }}>{foundationalKnowledge.explanationStrategy}</Typography>
          </CardContent>
        </Card>

        {/* 面接対策: 想定質問と回答ヒント（Q&A吹き出し風） */}
        <Card sx={{ mb: 3, bgcolor: '#fce4ec' }}>
          <CardContent>
            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
              <Avatar sx={{ bgcolor: '#d81b60', width: 32, height: 32 }}>
                <QuestionAnswerIcon />
              </Avatar>
              <Typography variant="h6" sx={{ fontWeight: 'bold' }}>面接対策: 想定質問と回答ヒント</Typography>
            </Stack>
            <Stack spacing={2}>
              {expectedQuestions.map((q, idx) => (
                <Box key={idx}>
                  {/* Q 吹き出し */}
                  <Box sx={{ display: 'flex', mb: 0.5 }}>
                    <Box sx={{ bgcolor: '#fff', border: '1px solid #d81b60', borderRadius: 2, px: 2, py: 1, maxWidth: '80%', fontWeight: 'bold', position: 'relative', ml: 0 }}>
                      <span style={{ color: '#d81b60' }}>Q.</span> {q.question}
                      <Box sx={{
                        content: '""',
                        position: 'absolute',
                        left: 16,
                        top: '100%',
                        width: 0,
                        height: 0,
                        borderLeft: '8px solid transparent',
                        borderRight: '8px solid transparent',
                        borderTop: '8px solid #d81b60',
                      }} />
                    </Box>
                  </Box>
                  {/* A 吹き出し */}
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 0.5 }}>
                    <Box sx={{ bgcolor: '#d81b60', color: '#fff', borderRadius: 2, px: 2, py: 1, maxWidth: '80%', fontWeight: 'bold', position: 'relative', mr: 0 }}>
                      <span style={{ color: '#fff' }}>A.</span> {q.answerTips}
                      <Box sx={{
                        content: '""',
                        position: 'absolute',
                        right: 16,
                        top: '100%',
                        width: 0,
                        height: 0,
                        borderLeft: '8px solid transparent',
                        borderRight: '8px solid transparent',
                        borderTop: '8px solid #d81b60',
                      }} />
                    </Box>
                  </Box>
                </Box>
              ))}
            </Stack>
          </CardContent>
        </Card>
      </Box>
    );
  }

  // パースできなかった場合は従来通りJSON表示
  const renderResult = (data) => {
    if (typeof data === "object" && data !== null) {
      return (
        <Box component="pre" sx={{ whiteSpace: "pre-wrap", wordBreak: "break-all", fontFamily: "monospace", fontSize: 15, p: 1 }}>
          {JSON.stringify(data, null, 2)}
        </Box>
      );
    }
    return <Typography>{data}</Typography>;
  };

  return (
    <Paper elevation={2} sx={{ p: 3, mt: 3, bgcolor: "#f8f9fa" }}>
      <Typography variant="h6" sx={{ mb: 2 }}>Gemini API レスポンス</Typography>
      {renderResult(result)}
    </Paper>
  );
};

export default GeminiResult;
