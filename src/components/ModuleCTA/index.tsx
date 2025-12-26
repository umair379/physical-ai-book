import React from 'react';
import styles from './styles.module.css';

interface ModuleCTAProps {
  moduleName: 'ROS2' | 'Gazebo' | 'Isaac' | 'VLA';
  moduleNumber: 1 | 2 | 3 | 4;
  moduleTitle: string;
  moduleUrl: string;
}

const MODULE_ICONS = {
  ROS2: '🤖',
  Gazebo: '🏗️',
  Isaac: '🧠',
  VLA: '👁️',
};

export default function ModuleCTA({
  moduleName,
  moduleNumber,
  moduleTitle,
  moduleUrl,
}: ModuleCTAProps): JSX.Element {
  return (
    <div className={styles.moduleCTA}>
      <div className={styles.ctaHeader}>
        <span className={styles.moduleIcon}>{MODULE_ICONS[moduleName]}</span>
        <span className={styles.moduleNumber}>Module {moduleNumber}</span>
      </div>
      <h3 className={styles.ctaTitle}>Continue Learning: {moduleTitle}</h3>
      <p className={styles.ctaDescription}>
        Explore the full module to dive deeper into {moduleName} concepts and hands-on exercises.
      </p>
      <a href={moduleUrl} className={styles.ctaButton}>
        Go to Module →
      </a>
    </div>
  );
}
