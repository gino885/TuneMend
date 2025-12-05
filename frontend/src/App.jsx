import React, { useState } from 'react';
import axios from 'axios';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend
} from 'recharts';
import {
  Container, Typography, TextField, Button, Box, Paper, Grid,
  CircularProgress, CssBaseline, ThemeProvider, createTheme,
  Card, IconButton, List, ListItem, ListItemText, Chip, Fade, Divider, useMediaQuery
} from '@mui/material';
import {
  AutoGraph, MusicNote, AddCircleOutline,
  DeleteOutline, QueueMusic, GraphicEq
} from '@mui/icons-material';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#00e5ff' },
    secondary: { main: '#f50057' },
    background: { default: '#050a14', paper: '#0f172a' },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h3: { fontWeight: 800, background: 'linear-gradient(135deg, #00C6FF 0%, #0072FF 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: 'rgba(15, 23, 42, 0.6)',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
          borderRadius: 24,
        }
      }
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 12,
            backgroundColor: 'rgba(0, 0, 0, 0.2)',
          }
        }
      }
    }
  }
});

const AppContent = () => {
  const isMobile = useMediaQuery(darkTheme.breakpoints.down('sm'));
  
  const [mood, setMood] = useState("");
  const [inputTitle, setInputTitle] = useState("");
  const [inputArtist, setInputArtist] = useState("");
  const [playlistQueue, setPlaylistQueue] = useState([]);
  const [result, setResult] = useState(null);
  const [isAdding, setIsAdding] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleAddSong = async () => {
    if (!inputTitle.trim()) return;
    setIsAdding(true);
    try {
      const response = await axios.post('http://localhost:8000/add_song', {
        title: inputTitle,
        artist: inputArtist || ""
      });
      if (!playlistQueue.some(s => s.title === response.data.title)) {
        setPlaylistQueue([...playlistQueue, response.data]);
        setInputTitle("");
        setInputArtist("");
      } else {
        alert("Song already in queue!");
      }
    } catch (error) {
      console.error(error);
      alert("Error finding song. Try adding artist name.");
    }
    setIsAdding(false);
  };

  const handleGenerate = async () => {
    if (!mood.trim()) return alert("Please enter your mood.");
    if (playlistQueue.length < 3) return alert("Need at least 3 songs.");
    setIsGenerating(true);
    try {
      const response = await axios.post('http://localhost:8000/generate_playlist', {
        user_mood: mood,
        songs: playlistQueue
      });
      setResult(response.data);
      setTimeout(() => document.getElementById("results")?.scrollIntoView({ behavior: 'smooth' }), 300);
    } catch (error) {
      alert("Generation failed.");
    }
    setIsGenerating(false);
  };

  return (
    <Container 
      maxWidth="lg" 
      sx={{ 
        minHeight: '100vh', 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center',     
        justifyContent: 'center', 
        py: 4,                   
      }}
    >

      {/* Header */}
      <Box textAlign="center" mb={isMobile ? 4 : 6} sx={{ width: '100%' }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ fontSize: { xs: '2.5rem', md: '3.5rem' }, letterSpacing: '-0.02em' }}>
          TuneMend
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ letterSpacing: '0.1em', textTransform: 'uppercase', fontSize: '0.75rem' }}>
          Biometric Emotion Resonance Engine
        </Typography>
      </Box>

      <Box sx={{ width: '100%', maxWidth: '600px', mb: result ? 8 : 0 }}>
        <Paper elevation={24} sx={{ p: isMobile ? 3 : 4 }}>

          {/* 1. Mood Input */}
          <Box mb={4}>
            <Box display="flex" alignItems="center" mb={1.5}>
              <GraphicEq color="secondary" sx={{ mr: 1 }} />
              <Typography variant="subtitle1" fontWeight="bold">Current Emotional State</Typography>
            </Box>
            <TextField
              fullWidth multiline rows={2}
              placeholder="I feel exhausted..."
              value={mood}
              onChange={(e) => setMood(e.target.value)}
            />
          </Box>

          <Divider sx={{ my: 3, borderColor: 'rgba(255,255,255,0.1)' }} />

          {/* 2. Add Songs */}
          <Box mb={3}>
            <Box display="flex" alignItems="center" mb={1.5}>
              <QueueMusic color="primary" sx={{ mr: 1 }} />
              <Typography variant="subtitle1" fontWeight="bold">Playlist Library</Typography>
            </Box>

            <Grid container spacing={1} mb={2} alignItems="stretch">
              <Grid item xs={12} sm={7}>
                <TextField
                  fullWidth size="small" placeholder="Song Title"
                  value={inputTitle} onChange={(e) => setInputTitle(e.target.value)}
                />
              </Grid>
              
              <Grid item xs={4} sm={2}>
                <Button
                  fullWidth variant="contained"
                  onClick={handleAddSong}
                  disabled={isAdding || !inputTitle}
                  sx={{ 
                    height: '100%', 
                    background: 'linear-gradient(45deg, #00e5ff, #0072ff)',
                    minWidth: 0 
                  }}
                >
                  {isAdding ? <CircularProgress size={20} color="inherit" /> : <AddCircleOutline />}
                </Button>
              </Grid>
            </Grid>

            {/* Song Queue List */}
            <Box sx={{ bgcolor: 'rgba(0,0,0,0.3)', borderRadius: 3, height: 150, overflowY: 'auto', mb: 2, p: 1 }}>
              {playlistQueue.length === 0 ? (
                <Box height="100%" display="flex" alignItems="center" justifyContent="center" color="text.disabled">
                  <Typography variant="caption">Queue is empty. Add 3+ songs.</Typography>
                </Box>
              ) : (
                <List dense>
                  {playlistQueue.map((song, idx) => (
                    <ListItem
                      key={idx}
                      secondaryAction={<IconButton size="small" onClick={() => setPlaylistQueue(playlistQueue.filter((_, i) => i !== idx))}><DeleteOutline fontSize="small" /></IconButton>}
                    >
                      <ListItemText
                        primary={song.title}
                        secondary={
                          <Typography variant="caption" color="text.secondary">
                            {song.artist || "Unknown Artist"}
                          </Typography>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              )}
            </Box>
          </Box>

          {/* Generate Button */}
          <Button
            fullWidth variant="contained" size="large"
            onClick={handleGenerate}
            disabled={isGenerating || playlistQueue.length < 3 || !mood}
            sx={{ py: 2, borderRadius: 3, fontSize: '1rem', background: 'linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%)', boxShadow: '0 4px 15px rgba(255, 65, 108, 0.3)' }}
          >
            {isGenerating ? "Synthesizing..." : "GENERATE HEALING PATH"}
          </Button>
        </Paper>
      </Box>

      {result && (
        <Fade in={true}>
          <Grid container spacing={4} id="results" sx={{ maxWidth: '1200px', mb: 8 }}>
            
            {/* Chart */}
            <Grid item xs={12} md={7}>
              <Paper sx={{ p: 3, height: 450, display: 'flex', flexDirection: 'column' }}>
                <Typography variant="h6" gutterBottom color="primary"><AutoGraph sx={{ mr: 1, verticalAlign: 'middle' }} />Affective Map</Typography>
                <Box flex={1}>
                  <ResponsiveContainer>
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis type="number" dataKey="v" domain={[-1, 1]} stroke="#666" />
                      <YAxis type="number" dataKey="a" domain={[-1, 1]} stroke="#666" />
                      <ReferenceLine x={0} stroke="#444" /> <ReferenceLine y={0} stroke="#444" />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderRadius: 8 }} />
                      <Legend />
                      <Scatter name="Path" data={result.path} fill="#00e5ff" line shape="circle" />
                      <Scatter name="Your Songs" data={playlistQueue} fill="#64748b" shape="triangle" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </Box>
              </Paper>
            </Grid>

            {/* Playlist Result */}
            <Grid item xs={12} md={5}>
              <Paper sx={{ p: 3, height: 450, overflowY: 'auto' }}>
                <Typography variant="h6" gutterBottom color="secondary"><MusicNote sx={{ mr: 1, verticalAlign: 'middle' }} />Sonic Journey</Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
                  {result.playlist.map((item, idx) => (
                    <Card key={idx} sx={{ p: 2, display: 'flex', alignItems: 'center', bgcolor: 'rgba(255,255,255,0.03)' }}>
                      <Box sx={{ width: 32, height: 32, borderRadius: '50%', bgcolor: idx === 4 ? 'primary.main' : 'rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', mr: 2, fontWeight: 'bold' }}>{item.stage_number}</Box>
                      <Box>
                        <Typography variant="subtitle2" fontWeight="bold">{item.song.title}</Typography>
                        <Typography variant="caption" color="text.secondary">{item.song.artist}</Typography>
                      </Box>
                      <Box flexGrow={1} />
                      <Chip label={`Stage ${item.stage_number}`} size="small" variant="outlined" color={idx >= 3 ? "primary" : "default"} />
                    </Card>
                  ))}
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </Fade>
      )}

      {/* Footer */}
      <Box mt={4} textAlign="center" color="text.secondary" sx={{ width: '100%', pb: 4 }}>
        <Typography variant="caption">
          Copyright © 2023 TuneMend Inc.
        </Typography>
      </Box>

    </Container>
  );
};

const App = () => {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ position: 'fixed', top: '-20%', left: '20%', width: '600px', height: '600px', background: 'radial-gradient(circle, rgba(0,229,255,0.1) 0%, rgba(0,0,0,0) 70%)', zIndex: -1 }} />
      <AppContent />
    </ThemeProvider>
  );
};

export default App;