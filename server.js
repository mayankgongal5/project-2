const express = require("express");
const aiRoutes = require("./routes/aiRoutes");
const psRoutes = require("./routes/psRoutes");
const daaRoutes = require("./routes/daaRoutes");

const app = express();
const port = process.env.PORT || 3000;

// Register route modules
app.use("/ai", aiRoutes);
app.use("/ps", psRoutes);
app.use("/daa", daaRoutes);

// Default route
app.get("/", (req, res) => {
  res.send("NOT FOUND");
});

// Handle 404
app.use((req, res) => {
  res.status(404).send("Route not found");
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});

module.exports = app;
