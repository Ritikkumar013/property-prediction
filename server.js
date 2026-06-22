require("dotenv").config();

const express = require("express");
const cors = require("cors");

const propertyRoutes = require("./routes/property.routes");

const app = express();

app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.status(200).json({
    success: true,
    message: "Property Predictor API Running",
    version: "1.0.0",
    endpoints: {
      properties: "/api/properties",
      predict: "/api/properties/predict",
      compare: "/api/properties/compare",
      corridors: "/api/properties/corridors",
      corridorStats: "/api/properties/corridors/:corridor",
    },
  });
});

app.use("/api/properties", propertyRoutes);

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
