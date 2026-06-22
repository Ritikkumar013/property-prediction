const express = require("express");
const router = express.Router();

const {
  createProperty,
  getProperties,
  predict,
  compare,
  corridorStats,
  corridors,
} = require("../controllers/property.controller");

router.post("/", createProperty);
router.get("/", getProperties);

router.post("/predict", predict);
router.post("/compare", compare);

router.get("/corridors", corridors);
router.get("/corridors/:corridor", corridorStats);

module.exports = router;
