import gsap from 'gsap';
import SplitType from 'split-type';

// Split text into characters
function initTextSplit() {
  const title = new SplitType('.title', { types: 'chars' });
  return title;
}

// Theme toggle functionality
function initThemeToggle() {
  const html = document.documentElement;
  const themeToggle = document.querySelector('.theme-toggle');
  const headerText = document.querySelector('.header-text');
  const header = document.querySelector('.header');
  const navlinks = document.querySelectorAll('.container .header .nav-links .nav');

  const savedTheme = localStorage.getItem('theme') || 'light';
  html.dataset.theme = savedTheme;
  themeToggle.textContent = `${savedTheme === 'light' ? '🌙' : '☀️'}`;

  themeToggle.addEventListener('click', () => {
    const newTheme = html.dataset.theme === 'light' ? 'dark' : 'light';
    html.dataset.theme = newTheme;
    localStorage.setItem('theme', newTheme);
    themeToggle.textContent = `${newTheme === 'light' ? '🌙' : '☀️'}`;
    headerText.style.color = newTheme === 'light' ? '#000' : '#fff';
    header.style.backgroundColor = newTheme === 'light' ? 'rgb(0 0 0 / 5%)' : 'rgba(255, 255, 255, 0.05)';
    navlinks.style.color = newTheme === 'light' ? '#000' : '#fff';
    
  });
}



// Initial Animation
function initAnimation() {
  const tl = gsap.timeline();

  // Animate header
  tl.to('.header-text', {
    opacity: 1,
    y: 0,
    duration: 1,
    ease: 'power3.out'
  });

  // Animate title characters
  tl.to('.char', {
    opacity: 1,
    y: 0,
    duration: 1,
    stagger: 0.03,
    ease: 'power4.out'
  }, '-=0.5');

  // Animate blobs with more dynamic movement
  tl.to('.blob-1', {
    scale: 1,
    x: 0,
    y: 0,
    duration: 2,
    ease: 'elastic.out(1, 0.3)'
  }, '-=1');

  tl.to('.blob-2', {
    scale: 1,
    x: 50,
    y: 50,
    duration: 2,
    ease: 'elastic.out(1, 0.3)'
  }, '-=1.8');

  // Animate subtitle with gradient effect
  tl.to('.subtitle', {
    opacity: 1,
    y: 0,
    duration: 1,
    ease: 'power3.out'
  }, '-=1');
}

// Enhanced blob hover animation
function initBlobHover() {
  document.addEventListener('mousemove', (e) => {
    const { clientX, clientY } = e;
    const moveX = (clientX - window.innerWidth / 2) * 0.05;
    const moveY = (clientY - window.innerHeight / 2) * 0.05;

    gsap.to('.blob-1', {
      x: moveX * 1.5,
      y: moveY * 1.5,
      rotation: moveX * 0.05,
      duration: 1.5,
      ease: 'power2.out'
    });

    gsap.to('.blob-2', {
      x: moveX * 2 + 50,
      y: moveY * 2 + 50,
      rotation: -moveX * 0.05,
      duration: 1.5,
      ease: 'power2.out'
    });
  });
}

// Initialize everything
window.addEventListener('load', () => {
  initTextSplit();
  initThemeToggle();
  initAnimation();
  initBlobHover();
});