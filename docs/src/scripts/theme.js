const themeToggleBtn = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');
const themeText = document.getElementById('theme-text');

const updateButtonUI = (theme) => {
    if (theme === 'dark') {
        themeIcon.textContent = '☀️';
        themeText.textContent = 'Light Mode';
    } else {
        themeIcon.textContent = '🌙';
        themeText.textContent = 'Dark Mode';
    }
};

const currentTheme = document.documentElement.getAttribute('data-bs-theme');
if (themeToggleBtn) {
    updateButtonUI(currentTheme);

    themeToggleBtn.addEventListener('click', () => {
        const activeTheme = document.documentElement.getAttribute('data-bs-theme');
        const newTheme = activeTheme === 'dark' ? 'light' : 'dark';

        document.documentElement.setAttribute('data-bs-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateButtonUI(newTheme);
    });
}