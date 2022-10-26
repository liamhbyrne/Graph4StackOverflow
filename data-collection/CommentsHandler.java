import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

class CommentRow {
    private String id;
    private String postId;
    private String score;
    private String text;
    private String creationDate;
    private String userId;
    private String contentLicense;

    public CommentRow(String id, String postId, String score, String text,
                      String creationDate, String userId, String contentLicense) {
        this.id = id;
        this.postId = postId;
        this.score = score;
        this.text = text;
        this.creationDate = creationDate;
        this.userId = userId;
        this.contentLicense = contentLicense;
    }

    public String getId() {
        return id;
    }

    public String getPostId() {
        return postId;
    }

    public String getScore() {
        return score;
    }

    public String getText() {
        return text;
    }

    public String getCreationDate() {
        return creationDate;
    }

    public String getUserId() {
        return userId;
    }

    public String getContentLicense() {
        return contentLicense;
    }

    @Override
    public String toString() {
        return "CommentRow{" +
                "id='" + id + '\'' +
                ", postId='" + postId + '\'' +
                ", score='" + score + '\'' +
                ", text='" + text + '\'' +
                ", creationDate='" + creationDate + '\'' +
                ", userId='" + userId + '\'' +
                ", contentLicense='" + contentLicense + '\'' +
                '}';
    }
}
public class CommentsHandler extends DefaultHandler {
    private static final String ROW = "row";
    private static final String COMMENTS = "comments";
    private StringBuilder elementValue;

    @Override
    public void startElement(String uri, String lName, String qName, Attributes attr) throws SAXException {
        if (qName.equals(ROW)) {
            CommentRow commentRow = new CommentRow(
                attr.getValue("Id"),
                attr.getValue("PostId"),
                attr.getValue("Score"),
                attr.getValue("Text"),
                attr.getValue("CreationDate"),
                attr.getValue("UserId"),
                attr.getValue("ContentLicense")
            );
            System.out.println(commentRow);

        } else if (!qName.equals(COMMENTS)) {
            throw new SAXException(String.format("Unknown tag %s", qName));
        }

    }

    public static void main(String[] args) throws ParserConfigurationException, SAXException, IOException {
        final long start = System.currentTimeMillis();

        SAXParserFactory factory = SAXParserFactory.newInstance();
        SAXParser saxParser = factory.newSAXParser();
        CommentsHandler handler = new CommentsHandler();
        saxParser.parse(
                "..\\data\\raw\\stackoverflow.com-Comments\\comments.xml", handler);


        // time end
        final long durationInMilliseconds = System.currentTimeMillis()-start;
        System.out.println("SAX parse took " + durationInMilliseconds + "ms.");
    }
}
