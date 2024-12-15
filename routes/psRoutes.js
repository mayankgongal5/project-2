const express = require('express');
const router = express.Router();
const psResponses = require('../data/psResponses');

// PS routes handler
router.get('/:id', (req, res) => {
  const { id } = req.params;
  const response = psResponses[id];
  
  if (response) {
    res.send(response);
  } else {
    res.status(404).send('PS route not found');
  }
});

module.exports = router;