import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import ModuleCard, {type ModuleMetadata} from '@site/src/components/ModuleCard';

import styles from './index.module.css';

// Module metadata for Physical AI & Humanoid Robotics course
const MODULES: ModuleMetadata[] = [
  {
    moduleNumber: 1,
    title: 'ROS 2 Fundamentals',
    description: 'Learn the Robot Operating System 2 (ROS 2), the industry-standard framework for building robot applications. Master nodes, topics, services, and create your first robot control systems.',
    estimatedTime: '4 hours',
    chapters: [
      {chapterNumber: 1, title: 'Fundamentals', url: '/docs/module-1/chapter-1-fundamentals'},
      {chapterNumber: 2, title: 'Python Integration', url: '/docs/module-1/chapter-2-python-integration'},
      {chapterNumber: 3, title: 'URDF Modeling', url: '/docs/module-1/chapter-3-urdf-modeling'},
    ],
    indexPageUrl: '/docs/module-1/chapter-1-fundamentals',
  },
  {
    moduleNumber: 2,
    title: 'Digital Twin & Simulation',
    description: 'Build high-fidelity robot simulations in Isaac Sim. Learn to create digital twins of physical robots, simulate sensors, and test algorithms in a safe virtual environment.',
    estimatedTime: '5 hours',
    chapters: [
      {chapterNumber: 1, title: 'Gazebo Physics', url: '/docs/module-2/chapter-1-gazebo-physics'},
      {chapterNumber: 2, title: 'Unity Environments', url: '/docs/module-2/chapter-2-unity-environments'},
      {chapterNumber: 3, title: 'Sensor Simulation', url: '/docs/module-2/chapter-3-sensor-simulation'},
    ],
    indexPageUrl: '/docs/module-2/chapter-1-gazebo-physics',
  },
  {
    moduleNumber: 3,
    title: 'Isaac Brain: AI Perception & Navigation',
    description: 'Implement AI-powered robot perception using computer vision and deep learning. Integrate Isaac Sim, Isaac ROS, and Nav2 navigation to build autonomous navigation systems.',
    estimatedTime: '6 hours',
    chapters: [
      {chapterNumber: 1, title: 'Isaac Sim', url: '/docs/module-3/chapter-1-isaac-sim'},
      {chapterNumber: 2, title: 'Isaac ROS', url: '/docs/module-3/chapter-2-isaac-ros'},
      {chapterNumber: 3, title: 'Nav2 Navigation', url: '/docs/module-3/chapter-3-nav2'},
    ],
    indexPageUrl: '/docs/module-3/chapter-1-isaac-sim',
  },
  {
    moduleNumber: 4,
    title: 'Vision-Language-Action (VLA)',
    description: 'Build voice-controlled robots with LLM planning capabilities. Integrate speech recognition, natural language understanding, and multimodal AI models for human-robot interaction.',
    estimatedTime: '5 hours',
    chapters: [
      {chapterNumber: 1, title: 'Voice-to-Action Pipeline', url: '/docs/module-4/chapter-1-voice-to-action'},
      {chapterNumber: 2, title: 'LLM Planning & Reasoning', url: '/docs/module-4/chapter-2-llm-planning'},
      {chapterNumber: 3, title: 'Capstone: Full VLA Integration', url: '/docs/module-4/chapter-3-capstone'},
    ],
    indexPageUrl: '/docs/module-4/chapter-1-voice-to-action',
  },
];

// Quick links for resources
interface QuickLink {
  label: string;
  description: string;
  url: string;
  icon: string;
}

const QUICK_LINKS: QuickLink[] = [
  {
    label: 'Prerequisites & Setup',
    description: 'Install required tools: ROS 2, Isaac Sim, NVIDIA drivers, and Python packages.',
    url: '/docs/intro',
    icon: '🛠️',
  },
  {
    label: 'Getting Started Guide',
    description: 'Quick introduction to the course structure and learning path.',
    url: '/docs/intro',
    icon: '🚀',
  },
  {
    label: 'GitHub Repository',
    description: 'Access code examples, URDF files, and project templates.',
    url: 'https://github.com/yourusername/physical-ai-book',
    icon: '💻',
  },
  {
    label: 'AI Chatbot Assistant',
    description: 'Get instant answers to your robotics questions (Coming Soon).',
    url: '#',
    icon: '🤖',
  },
  {
    label: 'Community Discord',
    description: 'Join other learners and ask questions in our community.',
    url: '#',
    icon: '💬',
  },
];

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className={styles.heroTitle}>
          {siteConfig.title}
        </Heading>
        <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
        <p className={styles.heroDescription}>
          Build intelligent, autonomous humanoid robots from scratch. Learn ROS 2, Isaac Sim simulation,
          AI perception with YOLO, autonomous navigation, and voice-controlled LLM planning.
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/module-1/chapter-1-fundamentals">
            Start Learning →
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            View Prerequisites
          </Link>
        </div>
      </div>
    </header>
  );
}

function ModulesSection() {
  return (
    <section className={styles.modulesSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>
          Course Modules
        </Heading>
        <p className={styles.sectionDescription}>
          Master Physical AI & Humanoid Robotics through 4 comprehensive modules covering ROS 2,
          simulation, AI perception, and voice-controlled intelligence.
        </p>
        <div className={styles.modulesGrid}>
          {MODULES.map((module) => (
            <ModuleCard key={module.moduleNumber} module={module} />
          ))}
        </div>
      </div>
    </section>
  );
}

function QuickLinksSection() {
  return (
    <section className={styles.quickLinksSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>
          Quick Links & Resources
        </Heading>
        <div className={styles.quickLinksGrid}>
          {QUICK_LINKS.map((link) => (
            <Link
              key={link.label}
              to={link.url}
              className={styles.quickLinkCard}>
              <div className={styles.quickLinkIcon}>{link.icon}</div>
              <h3 className={styles.quickLinkTitle}>{link.label}</h3>
              <p className={styles.quickLinkDescription}>{link.description}</p>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Home"
      description="Learn to build intelligent, autonomous humanoid robots with ROS 2, Isaac Sim, AI perception, and voice-controlled LLM planning.">
      <HomepageHeader />
      <main>
        <ModulesSection />
        <QuickLinksSection />
      </main>
    </Layout>
  );
}
