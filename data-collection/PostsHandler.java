import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import javax.xml.XMLConstants;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import java.io.IOException;

class PostRow {
    private String Id;
    private String PostTypeId;
    private String AcceptedAnswerId;
    private String CreationDate;
    private String Score;
    private String ViewCount;
    private String Body;
    private String OwnerUserId;
    private String LastEditorUserId;
    private String LastEditorDisplayName;
    private String LastEditDate;
    private String Title;
    private String Tags;
    private String AnswerCount;
    private String CommentCount;
    private String FavoriteCount;
    private String CommunityOwnedDate;
    private String ContentLicense;

    public PostRow(String id, String postTypeId, String acceptedAnswerId,
                   String creationDate, String score, String viewCount,
                   String body, String ownerUserId, String lastEditorUserId,
                   String lastEditorDisplayName, String lastEditDate,
                   String title, String tags, String answerCount,
                   String commentCount, String favoriteCount,
                   String communityOwnedDate, String contentLicense) {
        Id = id;
        PostTypeId = postTypeId;
        AcceptedAnswerId = acceptedAnswerId;
        CreationDate = creationDate;
        Score = score;
        ViewCount = viewCount;
        Body = body;
        OwnerUserId = ownerUserId;
        LastEditorUserId = lastEditorUserId;
        LastEditorDisplayName = lastEditorDisplayName;
        LastEditDate = lastEditDate;
        Title = title;
        Tags = tags;
        AnswerCount = answerCount;
        CommentCount = commentCount;
        FavoriteCount = favoriteCount;
        CommunityOwnedDate = communityOwnedDate;
        ContentLicense = contentLicense;
    }

    @Override
    public String toString() {
        return "PostRow{" +
                "Id='" + Id + '\'' +
                ", PostTypeId='" + PostTypeId + '\'' +
                ", AcceptedAnswerId='" + AcceptedAnswerId + '\'' +
                ", CreationDate='" + CreationDate + '\'' +
                ", Score='" + Score + '\'' +
                ", ViewCount='" + ViewCount + '\'' +
                ", Body='" + Body + '\'' +
                ", OwnerUserId='" + OwnerUserId + '\'' +
                ", LastEditorUserId='" + LastEditorUserId + '\'' +
                ", LastEditorDisplayName='" + LastEditorDisplayName + '\'' +
                ", LastEditDate='" + LastEditDate + '\'' +
                ", Title='" + Title + '\'' +
                ", Tags='" + Tags + '\'' +
                ", AnswerCount='" + AnswerCount + '\'' +
                ", CommentCount='" + CommentCount + '\'' +
                ", FavoriteCount='" + FavoriteCount + '\'' +
                ", CommunityOwnedDate='" + CommunityOwnedDate + '\'' +
                ", ContentLicense='" + ContentLicense + '\'' +
                '}';
    }

    public String getId() {
        return Id;
    }

    public String getPostTypeId() {
        return PostTypeId;
    }

    public String getAcceptedAnswerId() {
        return AcceptedAnswerId;
    }

    public String getCreationDate() {
        return CreationDate;
    }

    public String getScore() {
        return Score;
    }

    public String getViewCount() {
        return ViewCount;
    }

    public String getBody() {
        return Body;
    }

    public String getOwnerUserId() {
        return OwnerUserId;
    }

    public String getLastEditorUserId() {
        return LastEditorUserId;
    }

    public String getLastEditorDisplayName() {
        return LastEditorDisplayName;
    }

    public String getLastEditDate() {
        return LastEditDate;
    }

    public String getTitle() {
        return Title;
    }

    public String getTags() {
        return Tags;
    }

    public String getAnswerCount() {
        return AnswerCount;
    }

    public String getCommentCount() {
        return CommentCount;
    }

    public String getFavoriteCount() {
        return FavoriteCount;
    }

    public String getCommunityOwnedDate() {
        return CommunityOwnedDate;
    }

    public String getContentLicense() {
        return ContentLicense;
    }
}

public class PostsHandler extends DefaultHandler {
    private static final String ROW = "row";
    private static final String POSTS = "posts";
    private StringBuilder elementValue;

    @Override
    public void startElement(String uri, String lName, String qName, Attributes attr) throws SAXException {
        if (qName.equals(ROW)) {
            PostRow postRow = new PostRow(
                    attr.getValue("Id"),
                    attr.getValue("PostTypeId"),
                    attr.getValue("AcceptedAnswerId"),
                    attr.getValue("CreationDate"),
                    attr.getValue("Score"),
                    attr.getValue("ViewCount"),
                    attr.getValue("Body"),
                    attr.getValue("OwnerUserId"),
                    attr.getValue("LastEditorUserId"),
                    attr.getValue("LastEditorDisplayName"),
                    attr.getValue("LastEditDate"),
                    attr.getValue("Title"),
                    attr.getValue("Tags"),
                    attr.getValue("AnswerCount"),
                    attr.getValue("CommentCount"),
                    attr.getValue("FavoriteCount"),
                    attr.getValue("CommunityOwnedDate"),
                    attr.getValue("ContentLicense")
            );
            System.out.println(postRow);

        } else if (!qName.equals(POSTS)) {
            throw new SAXException(String.format("Unknown tag %s", qName));
        }

    }

    public static void main(String[] args) throws ParserConfigurationException, SAXException, IOException {
        final long start = System.currentTimeMillis();

        SAXParserFactory factory = SAXParserFactory.newInstance();
        factory.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING,false);
        SAXParser saxParser = factory.newSAXParser();
        PostsHandler handler = new PostsHandler();
        saxParser.parse(
                "..\\data\\raw\\stackoverflow.com-Posts\\posts.xml", handler);


        // time end
        final long durationInMilliseconds = System.currentTimeMillis()-start;
        System.out.println("SAX parse took " + durationInMilliseconds + "ms.");
    }
}
