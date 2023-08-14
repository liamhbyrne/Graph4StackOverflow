import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.Configurator;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.sql.*;

public class SQLiteSession {
    private final Logger logger = LogManager.getLogger(SQLiteSession.class);
    private Connection conn;

    private PreparedStatement ps;

    private int maxBatchSize;
    private int currentBatchsize = 0;

    public SQLiteSession(String url) {
        Configurator.setLevel(logger.getName(), Level.WARN);
        try {
            // create a connection to the database
            conn = DriverManager.getConnection(url);
            conn.setAutoCommit(false);
            logger.info("Connection to SQLite has been established.");
        } catch (SQLException e) {
            logger.error(e.getMessage());
        }
    }

    public void endSession() {
        try {
            if (conn != null) {
                conn.close();
            }
        } catch (SQLException ex) {
            logger.error(ex.getMessage());
        }
    }

    public void createTables() throws SQLException, IOException {
        try (Statement stmt = conn.createStatement()) {
            String query = readSqlScript("create.sql");
            stmt.executeUpdate(query);
            System.out.println("Table creation query executed.");
            conn.commit();
        }
    }

    private static String readSqlScript(String fileName) throws IOException {
        try (InputStream inputStream = SQLiteSession.class.getResourceAsStream("/" + fileName);
             BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
            StringBuilder script = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                script.append(line).append("\n");
            }
            return script.toString();
        }
    }

    public void setPreparedStatement(String baseQuery, int batchSize) throws SQLException {
        this.ps = conn.prepareStatement(baseQuery);
        this.maxBatchSize = batchSize;
    }

    public PreparedStatement getPreparedStatement() {
        return ps;
    }

    public void processBatch() throws SQLException {
        ps.addBatch();
        if (++currentBatchsize > maxBatchSize) {
            ps.executeBatch();
            currentBatchsize = 0;
            logger.info("Executed batch insert.");
            commitChanges();
        }
    }

    public void commitChanges() throws SQLException {
        conn.commit();
    }

    public static void main(String[] args) throws SQLException, IOException {
        String db_address = args[0];  // "jdbc:sqlite:..\\stackoverflow.db"
        SQLiteSession s = new SQLiteSession(
                db_address
        );
        s.createTables();
        s.endSession();
    }
}