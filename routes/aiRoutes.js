const express = require('express');
const router = express.Router();
const aiResponses = require('../data/aiResponses');

// AI routes handler
router.get('/:id', (req, res) => {
  const { id } = req.params;
  const response = aiResponses[id];
  
  if (response) {
    res.send(response);
  } else {
    res.status(404).send('AI route not found');
  }
});

module.exports = router;