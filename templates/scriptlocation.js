document.getElementById('location').addEventListener('input', function (e) {
    const query = e.target.value.trim();
    
    if (query.length > 2) {  // Start fetching after 3 characters
      fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${query}`)
        .then(response => response.json())
        .then(data => {
          const suggestions = document.getElementById('suggestions');
          suggestions.innerHTML = ''; // Clear previous suggestions
          
          // Populate new suggestions
          data.forEach(location => {
            const listItem = document.createElement('li');
            listItem.textContent = location.display_name;
            listItem.addEventListener('click', () => {
              document.getElementById('location').value = location.display_name; // Populate input
              suggestions.innerHTML = ''; // Clear suggestions after selection
            });
            suggestions.appendChild(listItem);
          });
        })
        .catch(error => console.error('Error fetching location data:', error));
    } else {
      document.getElementById('suggestions').innerHTML = ''; // Clear suggestions if input is short
    }
  });
  