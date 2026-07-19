import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

export interface ChapterMetadata {
  chapterNumber: number;
  title: string;
  url: string;
}

export interface ModuleMetadata {
  moduleNumber: number;
  title: string;
  description: string;
  estimatedTime: string;
  chapters: ChapterMetadata[];
  indexPageUrl: string;
}

interface ModuleCardProps {
  module: ModuleMetadata;
}

export default function ModuleCard({module}: ModuleCardProps): ReactNode {
  return (
    <div className={styles.moduleCard}>
      <div className={styles.moduleHeader}>
        <span className={styles.moduleNumber}>Module {module.moduleNumber}</span>
        <span className={styles.estimatedTime}>{module.estimatedTime}</span>
      </div>

      <h3 className={styles.moduleTitle}>{module.title}</h3>

      <p className={styles.moduleDescription}>{module.description}</p>

      {module.chapters && module.chapters.length > 0 && (
        <div className={styles.chaptersList}>
          <h4 className={styles.chaptersTitle}>Chapters</h4>
          <ul className={styles.chaptersItems}>
            {module.chapters.map((chapter) => (
              <li key={chapter.chapterNumber}>
                <Link to={chapter.url} className={styles.chapterLink}>
                  {chapter.chapterNumber}. {chapter.title}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className={styles.moduleActions}>
        <Link
          className="button button--primary button--lg"
          to={module.indexPageUrl}>
          Start Module →
        </Link>
      </div>
    </div>
  );
}
