-- 已应用 ch07-ddl.sql 后执行一次；先备份业务库。新建测试库由 conftest 顺序执行。
SET NAMES utf8mb4;
ALTER TABLE conversations
  ADD COLUMN layer1_from_msg_id BIGINT UNSIGNED NULL COMMENT '此id之后为层1原文',
  ADD COLUMN layer2_tokens INT NOT NULL DEFAULT 0 COMMENT '完整checkpoint渲染后层2用量';
UPDATE conversations SET layer1_from_msg_id = summary_upto_msg_id
 WHERE summary_upto_msg_id IS NOT NULL;
CREATE TABLE IF NOT EXISTS conversation_summaries (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  conversation_id BIGINT UNSIGNED NOT NULL,
  seq INT NOT NULL,
  from_msg_id BIGINT UNSIGNED NOT NULL,
  upto_msg_id BIGINT UNSIGNED NOT NULL,
  content TEXT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_conv_seq (conversation_id, seq),
  KEY idx_conv_upto (conversation_id, upto_msg_id),
  FOREIGN KEY (conversation_id) REFERENCES conversations(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
-- 迁移旧滚动摘要；原始消息和checkpoint保持不变。
INSERT INTO conversation_summaries (conversation_id, seq, from_msg_id, upto_msg_id, content)
 SELECT id, 1, 1, summary_upto_msg_id, summary FROM conversations
 WHERE summary_upto_msg_id > 0 AND summary IS NOT NULL AND summary <> ''
 AND NOT EXISTS (SELECT 1 FROM conversation_summaries cs WHERE cs.conversation_id=conversations.id);
