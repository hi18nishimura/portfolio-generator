import React from "react";
import { Paper, List, ListItem, ListItemButton, ListItemText, IconButton, Box } from "@mui/material";
import MenuIcon from '@mui/icons-material/Menu';

const Sidebar = ({
  isSidebarOpen,
  sidebarRef,
  sidebarCollapsed,
  sidebarHovered,
  handleSidebarMouseEnter,
  handleSidebarMouseLeave,
  setSidebarCollapsed,
  selectedProject,
  handleProjectSelect,
  DUMMY_PROJECTS,
  NEW_PROJECT,
  isMobile
}) => {
  if (isMobile) return null;
  return (
    <Paper
      elevation={2}
      ref={sidebarRef}
      sx={{
        width: isSidebarOpen ? 260 : 56,
        minWidth: isSidebarOpen ? 260 : 56,
        bgcolor: "#fff",
        borderRadius: 0, 
        p: isSidebarOpen ? 2 : 0,
        boxShadow: "0 0 8px #eee",
        transition: "width 0.25s cubic-bezier(.4,0,.2,1)",
        position: "relative",
        zIndex: 2,
        display: "flex",
        flexDirection: "column",
        height: '100vh',
        overflow: 'auto',
      }}
      onMouseEnter={handleSidebarMouseEnter}
      onMouseLeave={handleSidebarMouseLeave}
    >
      <Box
        sx={{
          width: 56,
          height: 56,
          minWidth: 56,
          minHeight: 56,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          p: 0,
          m: 0,
        }}
      >
        <IconButton
          size="small"
          onClick={() => setSidebarCollapsed((prev) => !prev)}
          aria-label={isSidebarOpen ? "サイドバーを畳む" : "サイドバーを展開"}
          sx={{ m: 0, p: 0 }}
        >
          <MenuIcon />
        </IconButton>
      </Box>
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
  );
};

export default Sidebar;
